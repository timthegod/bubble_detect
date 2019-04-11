import json
from pprint import pprint
import math
with open('via_region_data.json') as f:
    data = json.load(f)

    for attr, val in data.items():
        for attr2, val2 in val.items():
            if attr2 == 'regions':
                for val3 in val2:
                    if val3['shape_attributes']['name'] == 'circle':
                        cx = val3['shape_attributes']['cx']
                        cy = val3['shape_attributes']['cy']
                        r = val3['shape_attributes']['r']
                        new_x = [int(cx+r*math.cos(i/180*math.pi)) for i in range(0,361,2)]
                        new_y = [int(cy+r*math.sin(i/180*math.pi)) for i in range(0,361,2)]
                        all_points_x = new_x
                        all_points_y = new_y
                        val3['shape_attributes']['cx'] = all_points_x
                        val3['shape_attributes']['cy'] = all_points_y

                        val3['shape_attributes']['all_points_x'] = val3['shape_attributes'].pop('cx')
                        val3['shape_attributes']['all_points_y'] = val3['shape_attributes'].pop('cy')
                        val3['shape_attributes']['name'] = 'polygon'

pprint(data)
with open('via_region_data.json', 'w') as f:
    json.dump(data, f)