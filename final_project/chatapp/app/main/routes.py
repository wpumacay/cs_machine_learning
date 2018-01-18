from flask import session, redirect, url_for, render_template, request
from . import main
from .forms import LoginForm


GLOBAL_ROOM_COUNT = 0

@main.route( '/', methods=['GET', 'POST'] )
def index():

    global GLOBAL_ROOM_COUNT

    form = LoginForm()

    if form.validate_on_submit() :

        GLOBAL_ROOM_COUNT += 1
        session['name'] = form.name.data
        session['room'] = '' + str( GLOBAL_ROOM_COUNT )
        print 'connected ( name, room ): ( ', session['name'], ', ' , session['room'], ' )'
        return redirect( url_for( '.appSelection' ) )

    elif request.method == 'GET' :

        form.name.data = session.get('name', '')

    return render_template( 'index.html', form = form )

@main.route( '/appSelection', methods=['GET', 'POST'] )
def appSelection():

    if request.method == 'GET' :
        return render_template( 'appSelection.html' )

    elif request.method == 'POST' :

        if request.form['submit'] == 'Chatbot' :
            return redirect( url_for( '.appChatbot' ) )
        elif request.form['submit'] == 'Translator' :
            return redirect( url_for( '.appTranslator' ) )
    
    return render_template( 'appSelection.html' )

########################################################################

# Chatbot routes

@main.route( '/appChatbot', methods=['GET', 'POST'] )
def appChatbot():
    if request.method == 'GET' :
        return render_template( 'appChatbotSelect.html' )
    
    elif request.method == 'POST' :
        if request.form['submit'] == 'chatbotData' :
            return redirect( url_for( '.appChatbotData' ) )
        elif request.form['submit'] == 'chatbotTest' :
            return redirect( url_for( '.appChatbotTest' ) )

@main.route( '/appChatbotData' )
def appChatbotData():
    return render_template( 'appChatbotGatherData.html' )

@main.route( '/appChatbotTest' )
def appChatbotTest():
    return render_template( 'appChatbotTest.html' )

########################################################################

########################################################################

# Translator routes

@main.route( '/appTranslator' )
def appTranslator() :
    return render_template( 'appTranslator.html' )

########################################################################