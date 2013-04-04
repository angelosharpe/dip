import re

### EMOTICONS ###
# facee moticon definitions
top = r'[<>O}\]3P]?'
eyes = r'[:=Xx;]'
tear = r'\'?'
nose = r'[oO-]?'
# d, D, ), ], /, S, (. [, \, |, p, P, o, O, c, C, @, 3, &
mouths = r'[dD\)\]/S\(\[\\\|pPoOcC@{}3&]'
sad_mouths = r'[/S\(\[\\\|cC@{]'
happy_mouths = r'[dD\)\]pP}3]'
other_mouths = r'[oO&]'

faces = ''.join([top, eyes, tear, nose, mouths])
sad_faces = ''.join([top, eyes, tear, nose, sad_mouths])
happy_faces = ''.join([top, eyes, tear, nose, happy_mouths])
other_faces = ''.join([top, eyes, tear, nose, other_mouths])

# other emoticons definitions
other_emoticons = [
    r'\^_+\^', r'o\.O', r'@_+@', r'-_+-',
    r'\.\.+', r',,+', r'<3',
]
other_sad_emoticons = [
    r'-_+-'
]
other_happy_emoticons = [
    r'\^_+\^', r'<3',
]
other_other_emoticons = [
    r'o\.O', r'@_+@', r'\.\.+', r',,+'
]
# regexps
emoticons_re = re.compile(
    '\s(' + '|'.join(other_emoticons + [faces]) + ')\s'
)
sad_emoticons_re = re.compile(
    '\s(' + '|'.join(other_sad_emoticons + [sad_faces]) + ')\s'
)
happy_emoticons_re = re.compile(
    '\s(' + '|'.join(other_happy_emoticons + [happy_faces]) + ')\s'
)
other_emoticons_re = re.compile(
    '\s(' + '|'.join(other_other_emoticons + [other_faces]) + ')\s'
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
tags_re = re.compile(r'(#\w+)|(@\w+)')


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
