import requests

def get_live_streams(api_key, region_code="US", max_results=50):
    """
    Fetches the top live streams currently trending on YouTube.

    Args:
        api_key (str): Your YouTube Data API v3 key.
        region_code (str, optional): Country code for regional trending streams (default: "US").
        max_results (int, optional): Number of results to fetch (default: 50).

    Returns:
        list: A list of YouTube live stream URLs.
    """

    URL = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "type": "video",
        "eventType": "live",
        "chart": "mostPopular",
        "regionCode": region_code,
        "maxResults": max_results,
        "key": api_key
    }

    try:
        response = requests.get(URL, params=params)
        response.raise_for_status()  # Raise an error if API call fails
        data = response.json()

        stream_urls = [
            f"https://www.youtube.com/watch?v={video['id']['videoId']}"
            for video in data.get("items", [])
        ]

        return stream_urls

    except requests.exceptions.RequestException as e:
        print(f"Error fetching live streams: {e}")
        return []
