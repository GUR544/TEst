# downloader_with_cookies.py
# An interactive script to download YouTube videos.
# It automatically detects and uses a cookies.txt file from the home directory.
# It also auto-installs its main dependency, yt-dlp.

import os
import subprocess
import shutil

def install_yt_dlp():
    """
    Checks if yt-dlp is installed. If not, installs it via pip.
    """
    if shutil.which('yt-dlp'):
        # No need to print this every time, keeps the interface clean.
        return True
    
    print("🚀 yt-dlp not found. Attempting to install it via pip...")
    try:
        subprocess.run(['pip', 'install', '-U', 'yt-dlp'], check=True, capture_output=True, text=True)
        print("✅ Successfully installed yt-dlp.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Failed to install yt-dlp.\nStderr: {e.stderr}")
        return False

def download_video(video_url, output_path="downloads"):
    """
    Downloads the specified YouTube video or playlist, using cookies if available.
    """
    print(f"\n▶️ Preparing download for URL: {video_url}")
    os.makedirs(output_path, exist_ok=True)
    
    # --- This is the new logic ---
    # Define the path to the cookies file in the Codespace home directory
    cookies_file_path = os.path.expanduser('~/cookies.txt')

    # Base command structure
    command = [
        'yt-dlp',
        '-o', f'{output_path}/%(title)s [%(id)s].%(ext)s',
        '--progress',
        '--merge-output-format', 'mp4'
    ]

    # Check if the cookies file exists and add it to the command
    if os.path.exists(cookies_file_path):
        print("🍪 Found cookies.txt. The download will be authenticated.")
        command.extend(['--cookies', cookies_file_path])
    else:
        print("ⓘ No cookies.txt found. Proceeding with a standard (unauthenticated) download.")
    # --- End of new logic ---

    # Add the URL to the command
    command.append(video_url)
    
    try:
        print(f"📂 Videos will be saved in the '{output_path}' directory.")
        subprocess.run(command, check=True)
        print("\n✅ Download completed successfully!")
    except subprocess.CalledProcessError:
        print("\n❌ An error occurred during the download.")
        print("Check the URL. If it's a private video, ensure your cookies.txt is valid and in the home directory.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

def main():
    """
    Main function to run the interactive download loop.
    """
    if not install_yt_dlp():
        return

    print("\n--- YouTube Interactive Downloader (with Cookie Support) ---")
    print("This script will automatically use 'cookies.txt' if it's in your home directory.")
    
    while True:
        try:
            user_url = input("\nEnter a YouTube URL to download (or type 'q' to quit): ").strip()

            if user_url.lower() in ['q', 'quit', 'exit']:
                print("👋 Exiting the downloader. Goodbye!")
                break
            
            if not user_url:
                continue

            download_video(user_url)

        except KeyboardInterrupt:
            print("\n👋 Exiting the downloader. Goodbye!")
            break

if __name__ == "__main__":
    main()
