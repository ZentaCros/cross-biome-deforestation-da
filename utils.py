import os
import datetime

LOG_FILE = "change_log.md"

def init_log():
    # Only write header if file doesn't exist to prevent wiping
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("# FADE-Net Pipeline: Change Log & Tracking\n\n")

def log_change(message, category="INFO"):
    """
    Logs every major code generation, architectural decision, and bug fix.
    """
    init_log()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"- **[{timestamp}] [{category.upper()}]** {message}\n"
    print(log_entry, end="")
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    init_log()
    log_change("Initialized FADE-Net project structure.", "SETUP")
