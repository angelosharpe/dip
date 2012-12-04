import re

### EMOTICONS ###
# facee moticon definitions
top = r'[<>O}\]3P]?'
eyes = r'[:=Xx;]'
tear = r'\'?'
nose = r'[oO-]?'
# d, D, ), ], /, S, (. [, \, |, p, P, o, O, c, C, @, 3, &
mouths = r'[dD\)\]/S\(\[\\\|pPoOcC@{}3&]'
faces = ''.join([top, eyes, tear, nose, mouths])
# other emoticons definitions
other_emoticons = [
    r'\^_+\^', r'o\.O', r'@_+@', r'-_+-',
    r'\.\.+', r',,+', r'<3',
]
emoticons_re = re.compile(
    '\s(' + '|'.join(other_emoticons + [faces]) + ')\s'
)


### URLs ###
urls_re = re.compile(
    r'(https?|s?ftp)(://)(www\.)?([\w\.-]+\.[a-zA-Z]{2,4})(:\d*)?' +
    r'(/[-_~$.+!*\'()\[\],;:@&=\?/~#%\w#]*)[^\.\,\)\(\s]' +
    r'|' +
    r'(www\.)([\w\d\.-]+\.[a-zA-Z]{2,4})(:\d*)?' +
    r'(/[-_~$.+!*\'()\[\],;:@&=\?/~#%\w#]*)[^\.\,\)\(\s]'
)

### emails ###
emails_re = re.compile(r'\s([\w_\.-]+)(@)([\w\.-]+)(\.)([a-zA-Z]{2,6})')


### tweet tags ###
tags_re = re.compile(r'(#\w+)')


### time ###
time_re = re.compile(r'([0-2][0-9])(:)([0-9]{2})(\s*)(am|pm)?', flags=re.IGNORECASE)


### date ###
date_re = re.compile(
    r'(\d{1,2})([./-])(\d{1,2})([./-])(\d{2,4})' +
    r'|' +
    r'(\d{1,2})([./\s-])' +
    r'(jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)' +
    r'(uary|ruary|ch|il|e|ly|ust|tember|ober|ember)?'
    r'(\s)(\d{1,4})', flags=re.IGNORECASE)

### All kind of numbers ###
### Embeded apostrophe ###
