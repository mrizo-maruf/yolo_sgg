import json

data = '/home/yehia/rizo/THUD_Robot/Real_Scenes/10L/static/Capture_1/Label/2D_Object_Detection/frame-000000.color.json'

with open(data, 'r', encoding='utf-8') as f:
	json_data = json.load(f)

print(list(json_data.keys()))

print("shapes:", json_data['shapes'])