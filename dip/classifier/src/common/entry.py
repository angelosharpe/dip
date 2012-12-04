import logging
import nltk
import re
# stemmer
from nltk.stem.snowball import EnglishStemmer
#url parser for feature extraction
from urlparse import urlparse
# import regexps for feature extraction
import regexps
# import feature classes
from feature import *

#DONE:
#   Features:
#       Emoticons
#       n-tuples
#       URL
#       email
#TODO:
#   stematization
#   features:
#       message length
#       number count
#       count of ilnesses in text (from db)
#       count of symptoms in text (from db)
#       others????

class Entry:
    # lemmatizer class member
    stmr = EnglishStemmer()
    # defines word count in dictionary tuples
    MAX_TOKEN_SIZE = 2

    def __init__(self, entry, language, id=None, guid=None, label=None):
        self._logger = logging.getLogger()
        # entry id in dabatase
        self.id = id
        self.guid = guid
        self.classified = None
        self.text = entry
        self.language = language
        self.label = label
        self.features_func = {
                'url':[
                    self._feature_url_whole,
                    self._feature_url_domain,
                    self._feature_url_y,
                    self._feature_url_y_n,
                    None],
                'email':[
                    self._feature_email_whole,
                    self._feature_email_y,
                    self._feature_email_y_n,
                    None],
                'emoticon':[
                    self._feature_emoticon,
                    None],
                'tag':[
                    self._feature_tag,
                    None],
                'sentence':[
                    self._feature_sentence_count,
                    None],
                'time':[
                    self._feature_time,
                    self._feature_time_24h,
                    self._feature_time_24h_hours_only,
                    None],
                'date':[
                    self._feature_date,
                    self._feature_date_formated_dmy,
                    self._feature_date_formated_my,
                    self._feature_date_formated_y,
                    None],
                }


    def _to_sentences(self, entry):
        ''''
        This method splits string into sentences according to language of
        the string. Other languages are also supported but not yet implemented.
        '''
        if not entry:
            return []
        if self.language == 'de':
            tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
        elif self.language == 'cs':
            tokenizer = nltk.data.load('tokenizers/punkt/czech.pickle')
        else:
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return tokenizer.tokenize(entry)

    def _to_words(self, text):
        '''
        This method splits text into sentences and those sentences into
        words. Sentences are split on nonaplha characters then empty words
        are removed. Then stematizer is used to simplyfy words used as tokens
        in classifier.
        @param text whole text (this is the last step of feature export)
        '''
        word_list = []
        sentences = self._to_sentences(text)
        for sentence in sentences:
            raw_words = re.split(r'\W+', sentence)
            words = [self.stmr.stem(word) for word in filter(None, raw_words)]
            word_list.append(words)
        return word_list

    def _get_ntuple_token(self):
        '''
        This method generates tokens(N-tuples) from word lists
        '''
        word_list = self._to_words(self.text)
        for sentence in word_list:
            for n in xrange(1, self.MAX_TOKEN_SIZE + 1):
                for i in xrange(len(sentence) - n + 1):
                    yield Ntuple(tuple(sentence[i:i+n]))

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

    def _get_sentence_count_token(self):
        '''
        This method yields only one token containing count of sentences in input
        text
        '''
        return str(len(self._to_sentences(self.text)))

    #FEATURES
    def _feature_url_whole(self):
        '''
        Yield URLs -- use whole url.
        '''
        for found_url in self._get_re_token(regexps.urls_re):
            full_url = ''.join(list(found_url))
            yield Url(full_url)

    def _feature_url_domain(self):
        '''
        Yield URLs -- use only domain names.
        '''
        for found_url in self._get_re_token(regexps.urls_re):
            full_url = ''.join(list(found_url))
            if re.match(r'^w', full_url):
                full_url = 'http://' + full_url
            url = urlparse(full_url).netloc
            if not url:
                url = full_url
            yield Url(url)

    def _feature_url_y(self):
        '''
        Yield emails -- is url present? Add token only if present.
        '''
        found = 0
        for found_url in self._get_re_token(regexps.urls_re):
            found = 1
            break
        if found:
            yield Url('YES')

    def _feature_url_y_n(self):
        '''
        Yield emails -- is url present? Add yes token if present and no it not.
        '''
        found = 0
        for found_url in self._get_re_token(regexps.urls_re):
            found = 1
            break
        if found:
            yield Url('YES')
        else:
            yield Url('NO')

    def _feature_email_whole(self):
        '''
        Yield emails -- use whole email.

        '''
        for email in self._get_re_token(regexps.emails_re):
            yield Email(''.join(list(email)))

    def _feature_email_y(self):
        '''
        Yield emails -- use whole email. Add token only if present.
        '''
        found = 0
        for email in self._get_re_token(regexps.emails_re):
            found = 1
            break
        if found:
            yield Email('YES')

    def _feature_email_y_n(self):
        '''
        Yield emails -- use whole email. Add yes token if present and no it not.
        '''
        found = 0
        for email in self._get_re_token(regexps.emails_re):
            found = 1
            break
        if found:
            yield Email('YES')
        else:
            yield Email('NO')

    def _feature_emoticon(self):
        '''
        Yield emoticon tokens.
        '''
        for emoticon in self._get_re_token(regexps.emoticons_re):
            yield Emoticon(''.join(list(emoticon)))

    def _feature_tag(self):
        '''
        Yield tag tokens.
        '''
        for tag in self._get_re_token(regexps.tags_re):
            yield Tag(''.join(list(tag)))

    def _feature_sentence_count(self):
        '''
        Yield sentence count.
        '''
        count = self._get_sentence_count_token()
        yield SentenceCount(count)

    def _feature_time(self):
        '''
        Yield time tokens.
        '''
        for time in self._get_re_token(regexps.time_re):
            yield Time(''.join(list(time)))

    def _get_24h_time(self, findall_output):
        '''
        This method coverts time from am/pm to 24h format
        '''
        hours = int(findall_output[0])
        minutes = int(findall_output[2])
        if findall_output[4] is not '':
            if re.match(r'pm|Pm|PM', findall_output[4]):
                hours += 12
        return (hours, minutes)

    def _feature_time_24h(self):
        '''
        Yield time tokens in 24h format.
        '''
        for time in self._get_re_token(regexps.time_re):
            formated_time = self._get_24h_time(time)
            yield Time('{0}:{1}'.format(*formated_time))

    def _feature_time_24h_hours_only(self):
        '''
        Yield time tokens in 24h format.
        '''
        for time in self._get_re_token(regexps.time_re):
            formated_time = self._get_24h_time(time)
            yield Time(formated_time[0])

    def _get_formated_date(self, findall_output):
        # dd.mm.yyyy or yy.mm.dd format
        if findall_output[0] is not u'':
            # yy.mm.dd
            if len(findall_output[4]) == 2:
                year = int('20' + findall_output[0])
                month = int(findall_output[2])
                day = int(findall_output[4])
            else:
                year = int(findall_output[4])
                month = int(findall_output[2])
                day = int(findall_output[0])
            # fix problem with swapped day/month
            if month > 12:
                month, day = day, month
        # dd month yyyy format
        else:
            month_to_num = {
                'jan':1,  'feb':2,  'mar':3,  'apr':4,  'may':5,  'jun':6,
                'jul':7,  'aug':8,  'sep':9,  'oct':10, 'nov':11, 'dec':12
            }
            day = int(findall_output[5])
            month = month_to_num[findall_output[7].lower()]
            year = findall_output[10]
        return (day, month, year)

    def _feature_date(self):
        '''
        Yield date tokens found in text
        '''
        for date in self._get_re_token(regexps.date_re):
            yield Date(''.join(list(date)))

    def _feature_date_formated_dmy(self):
        '''
        Yield date tokens converted into united format dd-mm-yyyy
        '''
        for date in self._get_re_token(regexps.date_re):
            formated_date = self._get_formated_date(date)
            yield Date('{0}-{1}-{2}'.format(*formated_date))

    def _feature_date_formated_my(self):
        '''
        Yield date tokens converted into united format mm-yyyy
        '''
        for date in self._get_re_token(regexps.date_re):
            formated_date = self._get_formated_date(date)
            yield Date('{1}-{2}'.format(*formated_date))

    def _feature_date_formated_y(self):
        '''
        Yield date tokens converted into united format yyyy
        '''
        for date in self._get_re_token(regexps.date_re):
            formated_date = self._get_formated_date(date)
            yield Date('{2}'.format(*formated_date))

    def get_token(self):
        '''
        This method yields all possible tokens - uses all features 
        The most complex tokens should be extracted and then deleted first
        (eg. URLs, emails, emoticons,....,n-tuples). Found not-word tokens
        should be removed by the _get function.
        @param n defines maximaln n-tuple size (pass to _get_ntuple_token)
        '''
        # yield features
        for feature_type in self.features_func:
            if self.features_func[feature_type][0]:
                for feature in self.features_func[feature_type][0]():
                    #print feature
                    yield feature

        # yield n-tuples
        for ntuple in self._get_ntuple_token():
            yield ntuple

    def get_id(self):
        return self.id

    def get_guid(self):
        return self.guid

    def get_language(self):
        return self.language

    def get_original_entry(self):
        return self.original_entry
