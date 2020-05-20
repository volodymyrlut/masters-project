import hashlib
import json
from datetime import datetime

def timestamp():
	now = datetime.now()
	timestamp = datetime.timestamp(now)
	return timestamp

def calculate_hash(cell):
    return hash(json.dumps(cell))