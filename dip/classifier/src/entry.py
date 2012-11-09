import logging
import nltk
import re

#DONE:
#   Features:
#       n-tuples
#TODO:
#   stematization
#   features:
#       Emoticons
#       Count of sentences
#       message length
#       number count
#       count of ilnesses in text (from db)
#       count of symptoms in text (from db)
#       is there a date in text?
#       is there URL in text?
#       are there some #tags?
#       does the message contain email address?
#       others????

class Entry:
    def __init__(self, id, guid, entry, language):
        self._logger = logging.getLogger()
	self.id = id # entry id in database
        self.guid = guid
	self.classified = None # classified determines, how was the entry classified, None - was not classified yet
	self.original_entry = entry # entry message
        self.word_list = self._to_words(entry, language) # basic one
        self.language = language


    def _to_sentences(self, entry, language):
	'This method splits string into sentences according to language of the string. Other languages are also supported but not yet implemented.'
        if not entry:
            return []
	if language == 'de':
	    tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
	elif language == 'cs':
	    tokenizer = nltk.data.load('tokenizers/punkt/czech.pickle')
	else:
	    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        
        return tokenizer.tokenize(entry)

    def _to_words(self, entry, language):
	'This method splits entry into sentences and those sentences into words. Very simple spliting on non-alphanumeric characters.'
	word_list = []
	sentences = self._to_sentences(entry, language)
	for sentence in sentences:
	    words = re.split(r'\W+', sentence)
	    words = filter(None, words) # remove empty strings from list
	    word_list.append(words)
	return word_list

    def get_token(self, n):
	'This method generates tokens(N-tuples) from word lists'
	for sentence in self.word_list:
	    for i in xrange(len(sentence) - n + 1):
		yield tuple(sentence[i:i+n])
                
    def get_id(self):
        return self.id
    
    def get_guid(self):
        return self.guid

    def get_language(self):
        return self.language

    def get_original_entry(self):
        return self.original_entry
