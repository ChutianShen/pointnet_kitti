import json

db = []

query_content_dict_index = {}
query_content_dict_index['id'] = 0
query_content_dict_index['last'] = 1
query_content_dict_index['first'] = 2
query_content_dict_index['location'] = 3
query_content_dict_index['active'] = 4

location_dict_index = {}
location_dict_index['city'] = 0
location_dict_index['state'] = 1
location_dict_index['postalCode'] = 3

def jsonQuery(order, content): # order is add, get or delete; content is the query
    if order is 'add':
        store_json(content)
    elif order is 'get':
        return return_json(content)
    elif order is 'delete':
        delete_json(content)
    else:
        return 'Wrong order!!!'


def store_json(content):
    db.append(content)

def return_json(content):
    res = []
    for entry in db:
        if match(entry, content):
            res.append(entry)
    return res

def delete_json(content):
    for entry in db:
        if match(entry, content):
            db.remove(entry)

def match(entry, content):
    for condition in content:
        key = condition.keys()[0]   # .keys()[0] get its key
        if key is "location":
            #print key
            for location_condition in condition[key]:
                #print location_condition
                location_condition_key = location_condition.keys()[0]
                print location_condition_key
                print location_dict_index[location_condition_key]
                print entry[3]["location"][location_dict_index[location_condition_key]][location_condition_key]
                if entry[3]["location"][location_dict_index[location_condition_key]][location_condition_key] != location_condition[location_condition_key]:
                    return False
        else:
            if entry[query_content_dict_index[key]][key] != condition[key]:
                return False
    return True



entry1 = [
        {"id": 1}, {"last": "Doe"}, {"first": "John"},
        {"location": [{"city": "Oakland"}, {"state": "CA"}, {"postalCode": "94607"}]},
        {"active": True}
         ]

query1 = [{"id": 1}]
query2 = [{"id": 2}]
query3 = [{"location": [{"city": "Oakland"}]}]
query4 = [{"location": [{"city": "LA"}]}]

def run():
    jsonQuery("add", entry1)
    for entry in db:
        print entry

    # print match(db[0], query1)
    #
    # print match(db[0], query3)
    #
    # print jsonQuery("get", query1)
    #
    # print jsonQuery("get", query2)
    print match(db[0], query3)
    print match(db[0], query4)


run()