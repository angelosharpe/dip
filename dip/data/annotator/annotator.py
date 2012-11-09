#!/usr/bin/env python

"""
Softwere used for annotation of articles
Author: Tomas Marek
Date: 1.11.2012
"""

import sys
import re
import sqlite3

# dictionary for translating answers
num2txt = {'-1':'?', '0':'n', '1':'y'}

def get_entry(cur):
    """yield entries from db"""
    cur.execute('select distinct lang, relevance, text, annotation from docs where (annotation is null)')
    relevant = cur.fetchall()
    for entry in relevant:
        yield entry

if __name__ == "__main__":
    # check input and output dbs
    if len(sys.argv) != 3:
        print "Select db files!!!", sys.argc
        sys.exit(1)

    # open DB
    conn = sqlite3.connect(sys.argv[1])
    cur = conn.cursor()

    # open target DB
    conn_dest = sqlite3.connect(sys.argv[2])
    cur_dest = conn_dest.cursor()

    # start annotation
    for entry in get_entry(cur):
        lang = entry[0]
        classification = entry[1]
        text = entry[2]

        # automatic classification from DB
        if classification > 0.5:
            auto = 1
        elif classification < 0.5:
            auto = 0
        else:
            auto = -1

        # print text
        print '############################################################'
        print 'text:'
        print '############################################################'
        print text

        # manage answer
        answer = raw_input('Relevant? (y/n/?/END))[' + num2txt[str(auto)] + ']: ')
        if answer == 'y':
            annotation = 1
        elif answer == 'n':
            annotation = 0
        elif answer == '?':
            annotation = -1
        elif answer == 'END':
            break
        else:
            annotation = auto

        # update database
        cur.execute('update docs set annotation=? where lang=? and text=?', (annotation, lang, text))

        # store in dest database
        cur_dest.execute('INSERT INTO docs VALUES(?, ?, ?, ?)', (lang, classification, text, annotation))

        #commit
        conn.commit()
        conn_dest.commit()

    # close connection
    conn.close()
    conn_dest.close()
