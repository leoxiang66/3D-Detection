# remember to modify setup.py (version and requirement)
rm -rf dist/ build/ *.egg-info
python setup.py sdist
twine upload dist/*