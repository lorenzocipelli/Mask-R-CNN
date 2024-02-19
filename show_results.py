from pprint import pprint
import json


with open('results.json', 'r') as json_file:
    json_object = json.load(json_file)

pprint(json_object)