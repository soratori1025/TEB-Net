import requests
import json
import os

def download_figshare(article_id, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Get article details from Figshare API
    api_url = f"https://api.figshare.com/v2/articles/{article_id}/files"
    print(f"Fetching metadata from {api_url}")
    
    response = requests.get(api_url)
    if response.status_code != 200:
        print(f"Failed to fetch: {response.status_code}")
        return
        
    files = response.json()
    for file_info in files:
        file_name = file_info['name']
        download_url = file_info['download_url']
        save_path = os.path.join(save_dir, file_name)
        
        print(f"Downloading {file_name} from {download_url}...")
        r = requests.get(download_url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    download_figshare("2486866", "data/goldsmith_data")
