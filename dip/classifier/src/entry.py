import logging
import nltk
import re
from nltk.stem.snowball import EnglishStemmer

# import regexps for feature extraction
import regexps

#DONE:
#   Features:
#       Emoticons
#       n-tuples
#       URL
#       email
#TODO:
#   stematization
#   features:
#       URL - basename? whole?
#       email - only presence or whole address?
#       Count of sentences
#       message length
#       number count
#       count of ilnesses in text (from db)
#       count of symptoms in text (from db)
#       is there a date in text?
#       are there some #tags?
#       does the message contain email address?
#       others????

class Entry:
    # lemmatizer class member
    stmr = EnglishStemmer()

    def __init__(self, id, guid, entry, language):
        self._logger = logging.getLogger()
        # entry id in dabatase
        self.id = id
        self.guid = guid
        self.classified = None
        self.text = entry
        self.language = language


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
            words = [self.stmr.stem(word) for word in filter(None, raw_words)]
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

    def _get_re_token_and_rm(self, token_re):
        '''
        This method returns yields tokens matched with regexp in the original
        text and removes every occurance of matched patterns. Findall returns
        tuple of re groups which have to be joined back into one to be removed
        from original text.
        @param token_re regexp which determines token
        '''
        tokens = re.findall(token_re, self.text)
        if tokens:
            for token in set(tokens):
                self.text = re.sub(re.escape(''.join(list(token))), '', self.text)
            for token in tokens:
                yield token

    def _get_re_token(self, token_re):
        '''
        This method returns yields tokens matched with regexp in the original
        text. Findall returns tuple of re groups.
        @param token_re regexp which determines token
        '''
        tokens = re.findall(token_re, self.text)
        if tokens:
            for token in tokens:
                yield token

    def _get_sentenc_count_token(self, language):
        '''
        This method yields only one token containing count of sentences in input
        text
        @param language language determines which tokenizer will be used
        '''
        return str(len(self._to_sentences(self.text, language)))


    def get_token(self, n, language):
        '''
        This method yields all possible tokens - uses all features 
        The most complex tokens should be extracted and then deleted first
        (eg. URLs, emails, emoticons,....,n-tuples). Found not-word tokens
        should be removed by the _get function.
        @param n defines maximaln n-tuple size (pass to _get_ntuple_token)
        @param language  language determines which tokenizer will be used
        '''
        # yield URLs
        for url in self._get_re_token_and_rm(regexps.urls_re):
            out = '###feature###url:' + ''.join(list(url))
            print out
            yield out

        # yield emails
        for email in self._get_re_token_and_rm(regexps.emails_re):
            out = '###feature###email:' + ''.join(list(email))
            print out
            yield out

        # yield emoticons
        for emoticon in self._get_re_token_and_rm(regexps.emoticons_re):
            out = '###feature###emoticon:' + ''.join(list(emoticon))
            print out
            yield out

        # yield twitter tags
        for tag in self._get_re_token(regexps.tags_re):
            out = '###feature###tag:' + ''.join(list(tag))
            print out
            yield out

        # yield sentence count
        count = self._get_sentenc_count_token(language)
        out = '###feature###sentence_count=' + count
        print out
        yield out

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
