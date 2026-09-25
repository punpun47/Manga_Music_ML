import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

def get_spotify_object():
    load_dotenv()

    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.environ.get("SPOTIFY_REDIRECT_URI")

    sp = spotipy.Spotify(auth_manager = SpotifyOAuth(
        client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, 
        scope="user-modify-playback-state user-read-playback-state"))

    return sp
