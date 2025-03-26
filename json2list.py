def json_to_list(jsonfile):
    #this function has returned list with json
    import json

    json_file = open(jsonfile, encoding="utf-8")
    json_str = json_file.read()
    try:
        json_data = json.loads(json_str)['data']
        texts = []
        for dct in json_data:
            texts.append(dct['text'])
        return texts
    except KeyError:
        return [""]