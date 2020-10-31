##
# Variables
##

ENV_NAME = env
ENV_ACT = . env/bin/activate;
PIP = $(ENV_NAME)/bin/pip
PY = $(ENV_NAME)/bin/python

##
# Targets
##

.PHONY: install
install:
	rm -rf $(ENV_NAME)
	virtualenv -p python3 $(ENV_NAME)
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
