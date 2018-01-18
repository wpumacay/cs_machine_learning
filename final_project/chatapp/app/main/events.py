from flask import session
from flask_socketio import emit, join_room, leave_room
from .. import socketio
import time
import json
import model

#################################################################################################

# Chatbot related events

class LSessionInfo :

    def __init__ ( self, room, name ) :

        self.room = room
        self.name = name
        self.logs = []

    def addLog( self, msg, agentId ) :

        self.logs.append( { 'agent' : agentId, 'msg' : msg } )

    def writeToFileSystem( self ) :

        if len( self.logs ) < 1 :
            return

        _time = ( str( time.time() ).split( '.' ) )[0]

        _filename = 'log_%(room)s_%(name)s_%(time)s.log' % { 'room' : self.room, 'name' : self.name, 'time' : _time }

        _file = open( _filename, 'w' )

        _lines = []
        for _log in self.logs :
            _lines.append( json.dumps( _log ) + '\n\r' )

        _file.writelines( _lines )
        _file.close()

        print 'Wrote logs to file system :D'


GLOBAL_LOGS = {}

@socketio.on('joinedRoom', namespace='/appChatbotData')
def joinedRoom( message ):
    """Sent by clients when they enter a room.
    A status message is broadcast to all people in the room."""
    room = session.get( 'room' )
    name = session.get( 'name' )
    join_room( room )

    global GLOBAL_LOGS

    if room in GLOBAL_LOGS :
        GLOBAL_LOGS[room].writeToFileSystem()

    GLOBAL_LOGS[room] = LSessionInfo( room, name )

    emit( 'status', {'msg': 'My kingdom for some data!!! :(. Thanks for writing something :D'}, room = room )

@socketio.on( 'textAgent', namespace='/appChatbotData' )
def textAgent( message ):

    room = session.get( 'room' )
    name = session.get( 'name' )

    global GLOBAL_LOGS

    if not room in GLOBAL_LOGS :
        GLOBAL_LOGS[room] = LSessionInfo( room, name )

    GLOBAL_LOGS[room].addLog( message['msg'], 'agent' + message['agent'] )

    emit( 'message', {'msg':  'Agent' + message['agent'] + ': ' + message['msg']}, room = room )

@socketio.on( 'saveLogs', namespace='/appChatbotData' )
def saveLogs( message ) :

    room = session.get( 'room' )
    name = session.get( 'name' )

    global GLOBAL_LOGS

    if room in GLOBAL_LOGS :
        GLOBAL_LOGS[room].writeToFileSystem()
        GLOBAL_LOGS[room] = LSessionInfo( room, name )

    emit( 'status', { 'msg': "logs saved!, thanks :')" } )


@socketio.on('leftRoom', namespace='/appChatbotData')
def leftRoom(message):
    room = session.get('room')

    global GLOBAL_LOGS

    if room in GLOBAL_LOGS :
        GLOBAL_LOGS[room].writeToFileSystem()
        GLOBAL_LOGS.pop( room, None )

    leave_room( room )

    emit('status', {'msg': session.get('name') + ' has left the room.'}, room=room)


#################################################################################################

#################################################################################################

# Translator related events


@socketio.on('joinedRoomTranslator', namespace='/appTranslator')
def joinedRoomTranslator( message ):

    room = session.get( 'room' )
    join_room( room )

    emit( 'status', {'msg': 'Hi, write something and I will translate it'}, room = room )

@socketio.on( 'textUser', namespace='/appTranslator' )
def textUser( message ):

    room = session.get( 'room' )

    _strTranslated = model.onTranslate( message['msg'] )

    print 'sending back: ', _strTranslated, ' - room: ', room

    emit( 'message', { 'msgIn': message['msg'], 'msgOut': _strTranslated }, room = room )

@socketio.on('leftRoomTranslator', namespace='/appTranslator')
def leftRoomTranslator( message ) :
    room = session.get('room')

    leave_room( room )

    emit('status', {'msg': session.get('name') + ' has left the room.'}, room=room)