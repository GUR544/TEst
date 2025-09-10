# interactive_downloader.py
# An interactive script to download YouTube videos or playlists.
# It asks the user for a link, downloads it, and then asks for another.
# It also auto-installs its main dependency, yt-dlp.

import os
import subprocess
import shutil

def install_yt_dlp():
    """
    Checks if yt-dlp is installed. If not, installs it via pip.
    This function only runs once at the start of the script.
    """
    if shutil.which('yt-dlp'):
        print("✅ yt-dlp is already installed.")
        return True
    
    print("🚀 yt-dlp not found. Attempting to install it via pip...")
    try:
        # Run pip install command
        subprocess.run(
            ['pip', 'install', '-U', 'yt-dlp'], 
            check=True, 
            capture_output=True, 
            text=True
        )
        print("✅ Successfully installed yt-dlp.")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error: Failed to install yt-dlp.")
        print(f"Stderr: {e.stderr}")
        print("Please try running 'pip install -U yt-dlp' manually in your terminal.")
        return False

def download_video(video_url, output_path="downloads"):
    """
    Downloads the specified YouTube video or playlist to the output path.
    """
    print(f"\n▶️ Starting download for URL: {video_url}")
    
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Construct the yt-dlp command to download and merge into an MP4
    command = [
        'yt-dlp',
        '-o', f'{output_path}/%(title)s [%(id)s].%(ext)s',
        '--progress',
        '--merge-output-format', 'mp4',
        video_url
    ]
    
    try:
        # Execute the command
        print(f"📂 Videos will be saved in the '{output_path}' directory.")
        subprocess.run(command, check=True)
        print("\n✅ Download completed successfully!")
    except subprocess.CalledProcessError:
        print("\n❌ An error occurred during the download process.")
        print("Please check if the URL is correct and the video is public.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

def main():
    """
    Main function to run the interactive download loop.
    """
    # First, ensure our key dependency is installed
    if not install_yt_dlp():
        return # Exit if installation failed

    # Welcome message and instructions
    print("\n--- YouTube Interactive Downloader ---")
    print("You can paste a video or a playlist URL to start downloading.")
    print("The downloaded files will appear in the 'downloads' folder in the file explorer on the left.")
    
    # Start the interactive loop
    while True:
        try:
            # Ask the user for input
            user_url = input("\nEnter a YouTube URL to download (or type 'q' to quit): ").strip()

            # Check if the user wants to quit
            if user_url.lower() in ['q', 'quit', 'exit']:
                print("👋 Exiting the downloader. Goodbye!")
                break
            
            # Check for empty input
            if not user_url:
                continue

            # If input is provided, start the download
            download_video(user_url)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n👋 Exiting the downloader. Goodbye!")
            break

if __name__ == "__main__":
    main()
