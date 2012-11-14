import logging
import pickle

class WordDictionary:
    '''Class contains probability values of all allready treined tkoens.'''

    def __init__(self, pickle_filename=None):
        self._logger = logging.getLogger()
        self.filename = pickle_filename
        self.words = {}

    def wipe(self):
        self.words = {}

    def load(self):
        try:
            filehandler = open(self.filename,'rb')
        except IOError:
            self._logger.warning('Pickle file ' + self.filename + \
                    ' does not exist, no previous word dictionaries loaded')
            return
        self.words = pickle.load(filehandler)

    def store(self):
        filehandler = open(self.filename, 'wb')
        pickle.dump(self.words, filehandler)

    def to_xml(self, filename):
        self._logger.info('Generatingdictionary XML sorted by probability to \
                file: ' + filename + '...')
        if specification:
            f = open(filename + '_' + specification + '.pickle', 'w')
        else:
            f = open(filename, 'w')

        print >> f, '<?xml version="1.0" encoding="UTF-8"?>'
        if specification:
            print >> f, '<document ' + 'specification="' + specification + '">'
        print >> f, '<!-- Probability value is only for convenience. \
                Its value is calculated from probability / count. -->'
        for language in self.words:
            print >> f, '   <bayes_dictionary language="' + language + '">'
            for token in sorted(self.words[language],
                    key=lambda x: (self.words[language][x]['weight'] / 
                        self.words[language][x]['count'],
                        self.words[language][x]['count']), reverse=True):
                # if specification is entered
                if specification:
                    contains = False
                    for word in token:
                        if word == specification:
                            contains = True
                            break
                    if not contains:
                        continue;
                info = self.words[language][token]
                probability = round((info['weight'] / info['count']) * 100, 2)
                print >> f, '       <token weight="' + str(info['weight']) + \
                        '" count="' + str(info['count']) + '" probability="' + \
                        str(probability) + '%">'
                for word in token:
                    print >> f, '           <word>' + word + '</word>'
                print >> f, '       </token>'
            print >> f, '   </bayes_dictionary>'
        print >> f, '</document>'
        f.close()
