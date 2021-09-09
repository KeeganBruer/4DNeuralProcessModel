import json

class CustomJsonEncoder(json.JSONEncoder):
    def iterencode(self, obj, _one_shot=False):
        if isinstance(obj, float):
            obj_str = format(obj, '.12f')
            count = 0
            deci = obj_str.split(".")[1]
            for i in range(len(deci)):
                count += 1
                if (deci[i] != "0"):
                    break
            if (count < 2):
                count = 2
            obj_str = obj_str.split(".")[0] + "." + obj_str.split(".")[1][:count]
            yield obj_str.strip()
        elif isinstance(obj, dict):
            last_index = len(obj) - 1
            yield '{\n'
            i = 0
            for key, value in obj.items():
                yield '\t"' + key + '": '

                for chunk in CustomJsonEncoder.iterencode(self, value):
                    if (chunk == "[\n"):
                        yield chunk
                    else:
                        yield "\t" + chunk
                if i != last_index:
                    yield ", "
                yield "\n"
                i+=1
            yield '}'
        elif isinstance(obj, list):
            last_index = len(obj) - 1
            yield "[\n"
            for i, o in enumerate(obj):
                chunks = ""
                for chunk in CustomJsonEncoder.iterencode(self, o):
                     chunks += chunk
                if i != last_index:
                    yield "\t" +chunks + ",\n"
                else:
                    yield "\t" + chunks + "\n"
            yield "]"
        else:
            for chunk in json.JSONEncoder.iterencode(self, obj):
                yield chunk