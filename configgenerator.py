import json

jsondata = {
    "DRCDdataset": "/home/u8611808/text-segmentation/data/DRCD",
    "CMRCdataset": "/home/u8611808/text-segmentation/data/CMRC",
}

with open('config.json', 'w') as f:
    json.dump(jsondata, f)
