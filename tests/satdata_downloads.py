def get_remote_last_modified(url):
    try:
        response = requests.head(url)
        if 'Last-Modified' in response.headers:
            return datetime.strptime(response.headers['Last-Modified'], "%a, %d %b %Y %H:%M:%S %Z")
        else:
            print("Last-Modified header not found.")
            return None
    except Exception as e:
        print(f"Error checking remote file: {e}")
        return None

def download_from_opendap(remote_url, local_path):
    filename = remote_url.split('/')[-1]
    local_path = os.path.join(local_dir, filename)
    
    # Check if local file exists
    if os.path.exists(local_path):
        local_time = datetime.fromtimestamp(os.path.getmtime(local_path))
        remote_time = get_remote_last_modified(remote_url)

        if remote_time and remote_time <= local_time:
            print(f"{filename} is current.")
            return

    try:
        print(f"Fetching {filename}...")
        dataset = Dataset(remote_url)
        dataset.set_auto_mask(False)  # optional: disables masked arrays
        dataset.close()

        # This example saves a reference to the remote file (not downloading all data)
        with open(local_path, 'w') as f:
            f.write(f"Remote data accessed at: {remote_url}\n")

        print("Download complete.")
    except Exception as e:
        print(f"Failed to download: {e}")