import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
REMOTE_URL = "https://github.com/mamunice069/skin_lesion_xai.git"


def run_git(args):
    """গিট কমান্ড রান করার ফাংশন"""
    print(f">>> git {' '.join(args)}")
    subprocess.run(["git"] + args, cwd=ROOT_DIR, check=True)


def upload_to_github():
    # ১. .gitignore তৈরি করা (PYCHARM_GUIDE_BN.md এবং README.md বাদ দেওয়া হয়েছে)
    gitignore_content = """# ইউজারের নির্দেশ অনুযায়ী এই ২টি ফাইল বাদ দেওয়া হলো
/PYCHARM_GUIDE_BN.md
/README.md
report/README.md

# ভার্চুয়াল এনভায়রনমেন্ট ও ক্যাশ
.venv/
__pycache__/
*.pyc
.pytest_cache/
.idea/
*.egg-info/

# ১০০ MB-এর বড় ফাইল ও ক্যাশ ছবি (GitHub লিমিট এড়াতে)
*.pt
*.pth
*.npz
data/processed/
outputs/report_overleaf.zip
"""
    (ROOT_DIR / ".gitignore").write_text(gitignore_content.strip() + "\n", encoding="utf-8")
    print(">>> .gitignore আপডেট করা হয়েছে।")

    # ২. গিট ইনিশিয়ালাইজ এবং আপনার নাম ও ইমেইল সেট করা
    run_git(["init"])
    run_git(["config", "user.name", "mamunice069"])
    run_git(["config", "user.email", "mamunice069@gmail.com"])
    run_git(["config", "core.autocrlf", "true"])

    # ৩. ফাইলগুলো অ্যাড এবং কমিট করা
    run_git(["add", "."])

    # কমিট করা (আগে কমিট হয়ে থাকলে স্কিপ করবে)
    commit_res = subprocess.run(["git", "commit", "-m", "Initial project upload"], cwd=ROOT_DIR)
    if commit_res.returncode == 0:
        print(">>> সফলভাবে ফাইলগুলো কমিট (Commit) হয়েছে!")

    run_git(["branch", "-M", "main"])

    # ৪. রিমোট লিংক সেট করা এবং গিটহাবে পুশ করা
    subprocess.run(["git", "remote", "remove", "origin"], cwd=ROOT_DIR, stderr=subprocess.DEVNULL)
    run_git(["remote", "add", "origin", REMOTE_URL])

    print(f"\n>>> গিটহাবে আপলোড হচ্ছে: {REMOTE_URL} ...")
    run_git(["push", "-u", "origin", "main", "--force"])
    print("\n>>> অভিনন্দন! প্রজেক্টটি সফলভাবে গিটহাবে আপলোড হয়েছে!")


if __name__ == "__main__":
    upload_to_github()