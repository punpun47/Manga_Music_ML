from Spotify_Authenticator import get_spotify_object

# initialize spotify object
sp = get_spotify_object()

# temporarily hardcode
mood_label = "crying"

# map mood to spotify query
mood_to_query = {
    "crying": "sad_acoustic",
    "fight_scene": "action"
}

# get query
query = mood_to_query[mood_label]

# search spotify for track using query
results = sp.search(q=query, type="track", limit=1)

#TESTING
print(results)