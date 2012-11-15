import re

### EMOTICONS ###
# facee moticon definitions
anger = '>?'
eyes = '[:=8X;]'
tear = '\'?'
nose = '[oO-]?'
# d, D, ), ], /, S, (. [, \, |, p, P, o, O
mouths = '[dD\)\]/S\(\[\\\|pPoO]'
# other emoticons definitions
other_emoticons = [
    '\^_+\^', 'o\.O', '@_@', '-_+-',
    '\.\.+', ',,+', '<3',
]
emoticons_re = re.compile(
    '|'.join(other_emoticons) + '|' +
    anger +
    eyes +
    tear +
    nose +
    mouths
)

### URLs ###
urls_re = r'\s(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?\s'


### emails ###
email_re = r'\s([a-z0-9_\.-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})\s'


### tweet tags ###
tags = r'#\w+'


### Punctuation ###
### Timelike ###
### All kind of numbers ###
### Embeded apostrophe ###
