<div align="center">

<pre>
██████╗░░█████╗░██████╗░░█████╗░██╗░░██╗  ███████╗██╗░░░██╗███████╗
██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║░░██║  ██╔════╝╚██╗░██╔╝██╔════╝
██████╔╝██║░░██║██████╔╝██║░░╚═╝███████║  █████╗░░░╚████╔╝░█████╗░░
██╔═══╝░██║░░██║██╔══██╗██║░░██╗██╔══██║  ██╔══╝░░░░╚██╔╝░░██╔══╝░░
██║░░░░░╚█████╔╝██║░░██║╚█████╔╝██║░░██║  ███████╗░░░██║░░░███████╗
╚═╝░░░░░░╚════╝░╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝  ╚══════╝░░░╚═╝░░░╚══════╝
</pre>

</div>

### Add a category in one line
To add a new category (e.g., `vip`):

1. Create a folder: `known_faces/vip/<person_name>/...`
2. (Optional) If you want special behavior for a new category later, update `_category_to_status()` in `tracking.py`.  
   **By default, only `bad` is treated as disallowed; everything else is allowed.**

### Steps to run

1. Start virtual environment
`py -m venv .venv` 
`.\.venv\Scripts\Activate`

2. Install requirements
`pip install -r requirements.txt`

3. Start project
`python main.py`

### Formatting and linting
ruff format .
ruff check --fix .