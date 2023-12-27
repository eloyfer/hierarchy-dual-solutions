import sqlite3

class ConfigList:

    def __init__(self, configs):
        if isinstance(configs, ConfigList):
            self.configs = configs.configs
        elif isinstance(configs, list):
            self.configs = configs
        else:
            raise ValueError

    def __conform__(self, protocol):
        if protocol is sqlite3.PrepareProtocol:
            return repr(self)

    def __repr__(self):
        configs_str = [str(conf).replace(' ','') for conf in self.configs ]
        return ';'.join(configs_str)

    @staticmethod
    def convert_from_sql(s):
        configs = s.split(b';')
        configs = [x.rstrip(b')').lstrip(b'(').split(b',') for x in configs]
        configs = [tuple(map(int,x)) for x in configs]
        return ConfigList(configs)


