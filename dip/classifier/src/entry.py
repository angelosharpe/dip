import logging
import nltk
import re

# import regexps for feature extraction
import regexps

#DONE:
#   Features:
#       Emoticons
#       n-tuples
#TODO:
#   stematization
#   features:
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
        # entry id in dabatase
        self.id = id
        self.guid = guid
        self.classified = None
        self.text = entry
        self.language = language

    def _get_emoticon_token(self):
        '''
        This method returns list of emoticons in original sentence and removes
        found patterns
        Regexps are stored in 'regexps.py'
        '''
        emoticons = re.findall(regexps.emoticons_re, self.text)
        if emoticons:
            for emoticon in set(emoticons):
                re.sub(re.escape(emoticon), '', self.text)
            for emoticon in emoticons:
                yield emoticon

    def _to_sentences(self, entry, language):
        ''''
        This method splits string into sentences according to language of
        the string. Other languages are also supported but not yet implemented.
        '''
        if not entry:
            return []
        if language == 'de':
            tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
        elif language == 'cs':
            tokenizer = nltk.data.load('tokenizers/punkt/czech.pickle')
        else:
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return tokenizer.tokenize(entry)

    # TODO: stemming is soooooo inefficient
    def _to_words(self, text, language):
        '''
        This method splits text into sentences and those sentences into
        words. Sentences are split on nonaplha characters then empty words
        are removed. Then stematizer is used to simplyfy words used as tokens
        in classifier.
        @param text whole text (this is the last step of feature export)
        @param language language to pass to nltk tokenizer
        '''
        word_list = []
        sentences = self._to_sentences(text, language)
        for sentence in sentences:
            raw_words = re.split(r'\W+', sentence)
            #words = [common.st.stem(word) for word in filter(None, raw_words)]
            words = [word for word in filter(None, raw_words)]
            word_list.append(words)
        return word_list

    def _get_ntuple_token(self, n, language):
        '''
        This method generates tokens(N-tuples) from word lists
        @param n defines maximaln n-tuple size
        @param language language determines which tokenizer will be used
        '''
        word_list = self._to_words(self.text, language)
        for sentence in word_list:
            for i in xrange(len(sentence) - n + 1):
                yield tuple(sentence[i:i+n])

    def get_token(self, n, language):
        '''
        This method yields all possible tokens - uses all features 
        The most complex tokens should be extracted and then deleted first
        (eg. URLs, emails, emoticons,....,n-tuples). Found not-word tokens
        should be removed by the _get function.
        @param n defines maximaln n-tuple size (pass to _get_ntuple_token)
        @param language  language determines which tokenizer will be used
        '''
        # yield emoticons
        for emoticon in self._get_emoticon_token():
            yield emoticon

        # yield n-tuples
        for ntuple in self._get_ntuple_token(n, language):
            yield ntuple

    def get_id(self):
        return self.id

    def get_guid(self):
        return self.guid

    def get_language(self):
        return self.language

    def get_original_entry(self):
        return self.original_entry
