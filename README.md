<div align="center">

<pre>
################################################################################
#                                                                              #
#                                C R O W D   V I S I O N                       #
#                                                                              #
################################################################################
</pre>

</div>

### Steps to run

# 1) Start virtual environment
py -m venv .venv
.\.venv\Scripts\Activate

# 2) Ensure pip targets this interpreter
python -m pip install --upgrade pip setuptools wheel

# 3) Install DEV + base (dev file includes base)
python -m pip install -r .\requirements-dev.txt

3. Start project
`python main.py`

### Formatting and linting
ruff format .
ruff check --fix .