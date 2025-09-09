#!/usr/bin/env python3

import subprocess
import shutil
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress
from rich.panel import Panel

# Initialize Rich Console
console = Console()


def check_codespace_dependencies():
    """Checks if required command-line tools for Codespace are installed."""
    console.print("[bold cyan]Checking Codespace dependencies...[/bold cyan]")
    dependencies = ["gh", "yt-dlp"]
    missing_deps = [dep for dep in dependencies if shutil.which(dep) is None]

    if missing_deps:
        console.print(f"[bold red]Error: The following dependencies are not installed: {', '.join(missing_deps)}[/bold red]")
        console.print("Please ensure they are available in your Codespace environment.")
        exit(1)
    console.print("[bold green]Dependencies are satisfied.[/bold green]\n")


def github_auth():
    """Prompts the user to login to GitHub using the GitHub CLI."""
    console.print("[bold cyan]Authenticating with GitHub...[/bold cyan]")
    try:
        # Check current auth status to avoid unnecessary logins
        subprocess.run(["gh", "auth", "status"], check=True, capture_output=True, text=True)
        console.print("[green]Already logged into GitHub.[/green]\n")
    except subprocess.CalledProcessError:
        console.print("[yellow]Not logged in. Starting GitHub authentication process...[/yellow]")
        try:
            subprocess.run(["gh", "auth", "login"], check=True)
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]An error occurred during GitHub authentication: {e}[/bold red]")
            exit(1)


def create_upload_directory():
    """Creates the 'upload' directory in the home directory."""
    (Path.home() / "upload").mkdir(exist_ok=True)


def download_video(url: str):
    """Downloads a YouTube video to the 'upload' directory."""
    upload_dir = Path.home() / "upload"
    console.print(f"[bold cyan]Downloading video to {upload_dir}...[/bold cyan]")

    try:
        # Popen allows the script to continue to the progress bar.
        # Added '--no-progress' to prevent yt-dlp's output from cluttering the console.
        process = subprocess.Popen(
            ["yt-dlp", "-P", str(upload_dir), url, "--no-progress"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        with Progress() as progress:
            # The total=None makes this an indeterminate spinner.
            progress.add_task("[green]Downloading...", total=None)
            
            # Wait for the download process to complete.
            # The spinner will be active on screen during this time.
            process.wait()

        # Check for errors after the process has finished.
        if process.returncode != 0:
            stderr_output = process.stderr.read()
            console.print(f"[bold red]Error downloading video:\n{stderr_output}[/bold red]")
            exit(1)
            
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        exit(1)


def display_transfer_instructions():
    """Displays instructions for transferring the file to Termux."""
    instructions = """
[bold green]✅ Download Complete in Codespace![/bold green]

Your video has been downloaded to the 'upload' folder.

[bold]Next Steps:[/bold]
1. Open a [bold cyan]NEW[/bold cyan] terminal session in [bold]Termux[/bold] on your phone.
2. Make sure you have the GitHub CLI installed in Termux:
   [white]`pkg install gh`[/white]
3. Run the following command in Termux to securely copy the files:
   [white on black] gh codespace scp -r 'upload/*' ~/upload/ [/white on black]
   (You may be asked to log in to GitHub in Termux the first time).
4. Once the transfer is complete, run the processing script in Termux:
   [white]`python process_in_termux.py`[/white]
"""
    console.print(Panel.fit(instructions, title="[bold yellow]File Transfer Instructions[/bold yellow]", border_style="blue"))


def main():
    """Main function to run the Codespace downloader."""
    check_codespace_dependencies()
    github_auth()
    create_upload_directory()

    video_url = Prompt.ask("[bold]Enter the YouTube long video URL[/bold]")
    download_video(video_url)
    
    display_transfer_instructions()


if __name__ == "__main__":
    main()
