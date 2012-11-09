import logging

class Tests:
    'Class contains result forom tests. Printing this class prints nice output.'
    def __init__(self):
        self._logger = logging.getLogger()
	self.test_len = None
	self.train_len = None
	self.train_positive_len = None
	self.train_negative_len = None
	self.corelation = None
	self.percents = None

    def __str__(self):
	ret = 'TEST RESULTS:\n'
	ret += 'Training set size = ' + str(self.train_len) + '(' + str(self.train_positive_len) + ' relevant,' + str(self.train_negative_len) + ' irelevant)\n'
	ret += 'Test set size = ' + str(self.test_len) + '(' + str(self.percents[1] + self.percents[4]) + ' relevant,' + str(self.percents[2] + self.percents[3]) + ' irelevant)\n'
	ret += '-------------------------------------\n'
	if not self.corelation is None:
	    ret += 'Corelation test:\n'
	    ret += 'Result is in <-1, 1>, where -1 is worst and 1 is the best corelation.\n'
	    ret += 'Corelation = ' + str(self.corelation) + '\n'
	    ret += '-------------------------------------\n'
	if not self.percents is None:
	    ret += 'Percentage classification of accuracy:\n'
	    ret += 'True positive = ' + str(self.percents[1]) + ' (' + str(round((100 * self.percents[1]) / self.percents[6], 2)) + '%)\n'
	    ret += 'True negative = ' + str(self.percents[2]) + ' (' + str(round((100 * self.percents[2]) / self.percents[6], 2)) + '%)\n'
	    ret += 'False positive = ' + str(self.percents[3]) + ' (' + str(round((100 * self.percents[3]) / self.percents[6], 2)) + '%)\n'
	    ret += 'False negative = ' + str(self.percents[4]) + ' (' + str(round((100 * self.percents[4]) / self.percents[6], 2)) + '%)\n'
	    ret += 'Unknown = ' + str(self.percents[5]) + ' (' + str(round((100 * self.percents[5]) / self.percents[6], 2)) + '%)\n'
            ret += 'Precision = ' + str(self.percents[1] / (self.percents[1] + self.percents[3])) + '\n'
            ret += 'Recall = ' + str(self.percents[1] / (self.percents[1] + self.percents[4])) + '\n'
            ret += 'Accuracy = ' + str((self.percents[1] + self.percents[2]) / (self.percents[1] + self.percents[2] + self.percents[3] + self.percents[4])) + '\n'
	    ret += '-------------------------------------\n'
	return ret

    def set_corelation(self, corelation):
	self.corelation = corelation

    def set_percents(self, percents):
	self.percents = percents

    def set_test_len(self, length):
	self.test_len = length

    def set_train_len(self, length):
	self.train_len = length

    def set_train_positive_len(self, length):
	self.train_positive_len = length

    def set_train_negative_len(self, length):
	self.train_negative_len = length

    def get_percents(self):
        return self.percents

    def get_corelation(self):
        return self.corelation
