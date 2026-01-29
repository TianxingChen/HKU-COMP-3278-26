<h1 align="center">HKU Comp 3278'26</h1>

> Here is a tutorial for your group project. For this demo, we use Vanna with Deepseek for Text-to-SQL Generation, ChatGPT for UI vibe coding. The tutorial slides can be found in this [link](./Group_Project.pdf).

## 1. Install git locally

> Minimal steps to install Git locally so you can use `git clone`.  
> Verify after installation: `git --version`.

<details>
<summary><strong>Windows (10/11)</strong></summary>

**Option A (Recommended): Git for Windows**
1. Download from **https://git-scm.com/download/win**
2. Run installer (default options are fine)
3. Verify in **Command Prompt / PowerShell**:
   - `git --version`

**Notes**
- Includes **Git Bash** (recommended terminal for Git)
- Automatically sets up PATH

</details>

<details>
<summary><strong>macOS</strong></summary>

**Option A (Recommended): Homebrew**
1. Install Homebrew (if needed)
2. Install Git:
   - `brew install git`
3. Verify:
   - `git --version`

**Option B: Xcode Command Line Tools**
1. Run:
   - `git --version`
2. Follow the prompt to install Command Line Tools

</details>

<details>
<summary><strong>Ubuntu / Debian</strong></summary>

1. Update:
   - `sudo apt update`
2. Install:
   - `sudo apt install -y git`
3. Verify:
   - `git --version`

</details>

<details>
<summary><strong>Fedora / RHEL</strong></summary>

1. Install:
   - `sudo dnf install -y git`
   - (Older systems: `sudo yum install -y git`)
2. Verify:
   - `git --version`

</details>

<details>
<summary><strong>Arch / Manjaro</strong></summary>

1. Install:
   - `sudo pacman -S git`
2. Verify:
   - `git --version`

</details>

<details>
<summary><strong>Basic Usage</strong></summary>

1. Clone a repository:
   - `git clone <repo_url>`
2. Check status:
   - `git status`
3. Pull updates:
   - `git pull`

</details>

## 2. Clone this repo

Run the following code in your terminal.

```
git clone https://github.com/TianxingChen/HKU-COMP-3278-26.git
```

## 3. Install Python on Your Computer

> Minimal steps for the most common desktop OS.  
> After installing, verify in Terminal/Command Prompt: `python --version` (or `python3 --version`).

<details>
<summary><strong>Windows (10/11)</strong></summary>

**Option A (Recommended): Microsoft Store**
1. Open **Microsoft Store** → search **Python 3.x** → **Install**
2. Verify: `python --version`

**Option B: python.org installer**
1. Download from **python.org** (Windows installer)
2. Run installer → ✅ **Add Python to PATH** → Install
3. Verify: `python --version`

</details>

<details>
<summary><strong>macOS</strong></summary>

**Option A (Recommended): Homebrew**
1. Install Homebrew (if needed)
2. Install Python:
   - `brew install python`
3. Verify: `python3 --version`

**Option B: python.org installer**
1. Download macOS installer from **python.org**
2. Install → Verify: `python3 --version`

</details>

<details>
<summary><strong>Ubuntu / Debian</strong></summary>

1. Update:
   - `sudo apt update`
2. Install:
   - `sudo apt install -y python3 python3-pip python3-venv`
3. Verify: `python3 --version` and `pip3 --version`

</details>

<details>
<summary><strong>Fedora / RHEL</strong></summary>

1. Install:
   - Fedora: `sudo dnf install -y python3 python3-pip`
   - RHEL/CentOS: `sudo dnf install -y python3 python3-pip` (or `yum` on older systems)
2. Verify: `python3 --version`

</details>

<details>
<summary><strong>Arch / Manjaro</strong></summary>

1. Install:
   - `sudo pacman -S python python-pip`
2. Verify: `python --version`

</details>

<details>
<summary><strong>Create a Virtual Environment (All OS)</strong></summary>

1. Create:
   - `python -m venv .venv`  (use `python3` if needed)
2. Activate:
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`
3. Install packages:
   - `pip install <package>`
4. Deactivate:
   - `deactivate`

</details>


## 4. Install Python Environment

```
pip install -r requirements.txt
```

## 5. Register Deepseek API for Vanna

Deepseek API Platform: [https://platform.deepseek.com/usage](https://platform.deepseek.com/usage)

```
export DEEPSEEK_API_KEY=sk-xxx
```

## 6. Try Vanna

A text-to-SQL tutorial: [vanna.ai](link), we have installed the vanna locally in step 4.

Now please setup the database and Vanna by running:

```
python demo_chat_app.py
```


<details>
<summary><strong>Database and SQL Prompt (Interact with ChatGPT)</strong></summary>

```
I want a small, runnable Python demo that shows how “vibe coding” can turn an idea into a working system.

Please write a single-file FastAPI application that naturally blends two things together.

First, it should feel like a minimal group chat service:
- There are users and groups
- Users can join groups and post messages
- Messages are stored and can be queried later
- The goal is clarity and demonstrability, not a production-grade design

Second, it should also behave like an intelligent agent over the same data:
- It can directly reason over the chat data
- I can ask questions in natural language, such as
  “Which group is the most active?” or
  “Who has sent the most messages recently?”
- The agent figures out how to query the database on its own and returns answers

Overall style:
- Everything lives in a single file, so it’s easy to read and explain
- Use a lightweight database and prepare it automatically on startup
- Keep the code simple, readable, and well-structured
- It’s more important that it runs and tells a story than that it’s perfectly engineered

What I want students to take away is not the code itself,
but the feeling that:
“If you can clearly express the intent,
a system can almost grow itself.”
```
</details>

## 7. Visualize the Database

Use your web browser to open the `demo_index.html` (please keep `demo_chat_app.py` running).

<details>
<summary><strong>HTML Prompt (Interact with ChatGPT)</strong></summary>

I want a small HTML page for *looking into* a chat system, not chatting.
It should fetch real group names from the backend and let me browse messages safely and transparently.

</details>
