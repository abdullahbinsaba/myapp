import pyrebase

firebaseConfig = {
    "apiKey": "AIzaSyDeGoZ46Me02be1Al_V2eqt5ESsW3WoaHQ",
    "authDomain": "earthscape-auth.firebaseapp.com",
    "databaseURL": "https://earthscape-auth-default-rtdb.firebaseio.com",
    "projectId": "earthscape-auth",
    "storageBucket": "earthscape-auth.appspot.com",
    "messagingSenderId": "292462010928",
    "appId": "1:292462010928:web:2f0fed7a968bb454bc43f5",
    "measurementId": "G-71CQFQRB43"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()
