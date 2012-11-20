import re

### EMOTICONS ###
# facee moticon definitions
anger = r'>?'
eyes = r'[:=8X;]'
tear = r'\'?'
nose = r'[oO-]?'
# d, D, ), ], /, S, (. [, \, |, p, P, o, O
mouths = r'[dD\)\]/S\(\[\\\|pPoO]'
faces = ''.join([anger, eyes, tear, nose, mouths])
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
    r'\s(https?|s?ftp)(://)(www\.)?([\w\d]+\.[\w\d]{2,4}(:\d*)?)(/?\S*)\\n?'
)


### emails ###
emails_re = re.compile(r'\s([a-z0-9_\.-]+)(@)([\da-z\.-]+)(\.)([a-z\.]{2,6})\s')


### tweet tags ###
tags_re = re.compile(r'\s(#\w+)\s')


### Timelike ###
### All kind of numbers ###
### Embeded apostrophe ###
