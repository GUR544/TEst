from instagrapi import Client
import time
import os

USERNAME = "clutchzonegaming"
PASSWORD = "Sewak123@12"

# Store session data to avoid re-logging in repeatedly
# This also helps Instagram recognize your "device"
SESSION_FILE = "session.json"

cl = Client()

if os.path.exists(SESSION_FILE):
    cl.load_settings(SESSION_FILE)
    print("Loaded session from file.")

cl.delay_range = [3, 7] # Random delay for operations (seconds)

try:
    # If session loaded successfully, try to login with it
    cl.login(USERNAME, PASSWORD)
    print(f"Logged in as {USERNAME} using session.")

except Exception as e:
    print(f"Session login failed or not present: {e}")
    print("Attempting fresh login...")
    
    # Fresh login attempt
    try:
        cl.login(USERNAME, PASSWORD)
        print(f"Successfully logged in as {USERNAME}.")
        cl.dump_settings(SESSION_FILE) # Save new session

    except Exception as e_login:
        if "challenge_required" in str(e_login):
            print("Challenge required! Trying to solve it...")
            # This is a simplified challenge solver. 
            # Instagrapi has more advanced ways to handle this.
            # Often it involves sending a code via email/SMS.
            try:
                challenge_code = input("Enter challenge code sent to your email/SMS: ")
                cl.challenge_code_send(USERNAME, challenge_code) # or cl.challenge_solve()
                print("Challenge code submitted. Trying login again.")
                cl.login(USERNAME, PASSWORD) # Try login again after challenge
                print(f"Successfully logged in as {USERNAME} after challenge.")
                cl.dump_settings(SESSION_FILE) # Save new session

            except Exception as e_challenge:
                print(f"Failed to solve challenge or login after challenge: {e_challenge}")
                print("Please check your Instagram app for security alerts or try logging in manually.")
        elif "Two-factor authentication required" in str(e_login):
            print("2FA required!")
            twofa_code = input("Enter 2FA verification code: ")
            try:
                cl.two_factor_login(USERNAME, PASSWORD, twofa_code)
                print(f"Successfully logged in as {USERNAME} with 2FA.")
                cl.dump_settings(SESSION_FILE) # Save new session
            except Exception as e_2fa:
                print(f"Failed to login with 2FA: {e_2fa}")
                print("Please ensure the 2FA code is correct and try again.")
        else:
            print(f"Fatal login error: {e_login}")

if cl.is_logged_in:
    print("Bot is ready!")
    # You can now proceed with your liking logic using cl.hashtag_medias_top(), cl.user_info_by_username(), cl.media_like() etc.
    # Example:
    # media_ids = cl.hashtag_medias_top('pubgmobile', amount=10)
    # for media in media_ids:
    #    cl.media_like(media.pk)
    #    print(f"Liked {media.pk}")
    #    time.sleep(random.randint(5,10))
else:
    print("Login failed. Exiting.")
