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

class Entry:
    '''
    Entry object contains every text input for both classification and learning.
    It also provideas all operations on input text (stematization, features
    selection, etc.)
    '''
    # lemmatizer class member
    stmr = EnglishStemmer()

    def __init__(self, entry, language, id=None, guid=None, label=None, max_token_size=2):
        '''
        Init method
        @param entry: input text
        @param language: language of input text
        @param id: id from database
        @param guid: tweet guid from database
        @param label: label from manual classification
        @param max_token_size: text tokenization parameter
        '''
        self._logger = logging.getLogger()
        # entry id in dabatase
        self.id = id
        self.guid = guid
        self.classified = None
        self.text = entry
        self.language = language
        self.label = label
        self.MAX_TOKEN_SIZE=max_token_size
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
                    self._feature_emoticon_y,
                    self._feature_emoticon_y_n,
                    self._feature_emoticon_emotion,
                    None],
                'tag':[
                    self._feature_tag,
                    self._feature_tag_y,
                    self._feature_tag_y_n,
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
        self.features_func_count = dict([(x,list(xrange(len(self.features_func[x]))))
            for x in self.features_func])

    def check_feats(self, feats):
        '''
        Method checks whether input features selection dictionary is acceptable 
        by Entry class.
        @param feats: input features
        @return: boolean value True if feature dict contains all features in
                correct format
        '''
        # check number of features
        if len(feats) != len(self.features_func_count):
            message = 'Not all needed features were specified, please specify: '
            for feature in self.features_func_count:
                message += feature + ' '
            self._logger.error(message)
            return False
        # for every possible feature
        for feat in feats:
            # check if is correct
            if not feat in self.features_func_count:
                self._logger.error('Unknown feature "{0}"!!!'.format(feat))
                return False
            # check option number of this feature
            else:
                if feats[feat] >= len(self.features_func_count[feat]):
                    self._logger.error(
                        'Wrong feature type id ({0}) in feature "{1}". Max is {2}!!!'
                        .format(feats[feat], feat,
                            len(self.features_func_count[feat]) - 1))
                    return False
        return True

    def _to_sentences(self, entry):
        ''''
        This method splits string into sentences according to language of
        the string. Other languages are also supported but not yet implemented.
        @param entry: input text
        @return: list of sentences
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
        @param text: whole text (this is the last step of feature export)
        @return: list of words
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
        @return: yields text tokens
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
        @param token_re: regexp which determines token
        @return: yields text tokens
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
        @param token_re: regexp which determines token
        @return: yields text tokens
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
        @return: yields features
        '''
        for found_url in self._get_re_token(regexps.urls_re):
            full_url = ''.join(list(found_url))
            yield Url(full_url)

    def _feature_url_domain(self):
        '''
        Yield URLs -- use only domain names.
        @return: yields features
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
        @return: yields features
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
        @return: yields features
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
        @return: yields features
        '''
        for email in self._get_re_token(regexps.emails_re):
            yield Email(''.join(list(email)))

    def _feature_email_y(self):
        '''
        Yield emails -- Add token YES only if present.
        @return: yields features
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
        @return: yields features
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
        @return: yields features
        '''
        for emoticon in self._get_re_token(regexps.emoticons_re):
            yield Emoticon(''.join(list(emoticon)))

    def _feature_emoticon_y(self):
        '''
        Yield emoticon
        @return: yields features
        '''
        found = False
        for emoticon in self._get_re_token(regexps.emoticons_re):
            found = True
            break
        if found:
            yield Emoticon('YES')

    def _feature_emoticon_y_n(self):
        '''
        Yield emoticon
        @return: yields features
        '''
        found = False
        for emoticon in self._get_re_token(regexps.emoticons_re):
            found = True
            break
        if found:
            yield Emoticon('YES')
        else:
            yield Emoticon('NO')

    def _feature_emoticon_emotion(self):
        '''
        Yield emoticon emootion
        @return: yields features
        '''
        for sad_emoticon in self._get_re_token(regexps.sad_emoticons_re):
            yield Emoticon('SAD')
            break
        for happy_emoticon in self._get_re_token(regexps.happy_emoticons_re):
            yield Emoticon('HAPPY')
            break
        for other_emoticon in self._get_re_token(regexps.other_emoticons_re):
            yield Emoticon('OTHER')
            break


    def _feature_tag(self):
        '''
        Yield tag tokens.
        @return: yields features
        '''
        for tag in self._get_re_token(regexps.tags_re):
            yield Tag(''.join(list(tag)))

    def _feature_tag_y(self):
        '''
        Yield tag tokens.
        @return: yields features
        '''
        found = False
        for tag in self._get_re_token(regexps.tags_re):
            found = True
            break
        if found:
            yield Tag('YES')

    def _feature_tag_y_n(self):
        '''
        Yield tag tokens.
        @return: yields features
        '''
        found = False
        for tag in self._get_re_token(regexps.tags_re):
            found = True
            break
        if found:
            yield Tag('YES')
        else:
            yield Tag('NO')

    def _feature_sentence_count(self):
        '''
        Yield sentence count.
        @return: yields features
        '''
        count = self._get_sentence_count_token()
        yield SentenceCount(count)

    def _feature_time(self):
        '''
        Yield time tokens.
        @return: yields features
        '''
        for time in self._get_re_token(regexps.time_re):
            yield Time(''.join(list(time)))

    def _get_24h_time(self, findall_output):
        '''
        This method coverts time from am/pm to 24h format
        @return: yields features
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
        @return: yields features
        '''
        for time in self._get_re_token(regexps.time_re):
            formated_time = self._get_24h_time(time)
            yield Time('{0}:{1}'.format(*formated_time))

    def _feature_time_24h_hours_only(self):
        '''
        Yield time tokens in 24h format.
        @return: yields features
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
        @return: yields features
        '''
        for date in self._get_re_token(regexps.date_re):
            yield Date(''.join(list(date)))

    def _feature_date_formated_dmy(self):
        '''
        Yield date tokens converted into united format dd-mm-yyyy
        @return: yields features
        '''
        for date in self._get_re_token(regexps.date_re):
            formated_date = self._get_formated_date(date)
            yield Date('{0}-{1}-{2}'.format(*formated_date))

    def _feature_date_formated_my(self):
        '''
        Yield date tokens converted into united format mm-yyyy
        @return: yields features
        '''
        for date in self._get_re_token(regexps.date_re):
            formated_date = self._get_formated_date(date)
            yield Date('{1}-{2}'.format(*formated_date))

    def _feature_date_formated_y(self):
        '''
        Yield date tokens converted into united format yyyy
        @return: yields features
        '''
        for date in self._get_re_token(regexps.date_re):
            formated_date = self._get_formated_date(date)
            yield Date('{2}'.format(*formated_date))

    def get_token(self, features):
        '''
        This method yields selected tokens - features and n-tuples.
        @param features: dictionary containing chosen features and it's types
        @return: yields tokens
        '''
        # yield features
        for feature in features:
            if self.features_func[feature][features[feature]]:
                for f in self.features_func[feature][features[feature]]():
                    yield f

        # yield n-tuples
        for ntuple in self._get_ntuple_token():
            yield ntuple

    def get_token_all(self):
        '''
        This method yields all possible tokens - features and n-tuples.
        @return: yields tokens
        '''
        for feature in self.features_func_count:
            for i in self.features_func_count[feature]:
                if self.features_func[feature][i]:
                    for f in self.features_func[feature][i]():
                        yield f

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
