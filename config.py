'''Class to deal with the config file'''

import configparser
import copy

class Config():
    def __init__(self):
        self.parser = configparser.ConfigParser(inline_comment_prefixes='#')
        self.parser.read('config/config.ini')
        self.sections = self.parser.sections() 

        # dump config settings
        for section in self.sections:
            print("[",section,"]")
            for setting in self.parser.items(section):
                print(setting[0],"=",setting[1])
            print("")

    def sec(self,section):
        return dict(self.parser.items(section))

    def __repr__(self):
        '''Output when printing an instance of this class'''
        for sect in self.parser.sections():
          print('Section:', self.sect)
          for k,v in parser.items(sect):
             print(' {} = {}'.format(k,v))
          print()

    def __call__(self,section=None,par = None,dtype = None):
            '''Output when returning an instance of this class'''
            if section == None:
                return copy.deepcopy(self.parser.sections) 
            elif par is None:
                return dict(self.parser.items(section)) 
            elif dtype == 'int':
                return self.parser.getint(section,par)
            elif dtype == 'float':
                return self.parser.getfloat(section,par)
            elif dtype == 'bool':
                return self.parser.getboolean(section,par)
            elif dtype == 'str':
                return self.parser.get(section,par) 

    def update_config(self,section, par,val):
        self.parser.set(section,par,val)
