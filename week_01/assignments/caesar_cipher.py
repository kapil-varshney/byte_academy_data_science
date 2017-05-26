
def in_range(char):
	if ((ord(char) in range(65, 91)) or (ord(char) in range(97, 123))):
		return True
	else:
		return False

def letter_case(char) :
	if (ord(char) in range(65, 91)) :
		return 'upper'
	elif (ord(char) in range(97, 123)) :
		return 'lower'
	else :
		return 'unknown'

def cycle_through(char, shifted_char):
	
	c_code = ord(char)
	sc_code = ord(shifted_char)
	
	if ((c_code in range(65, 91)) and (sc_code < 65)):
		bound = 65
		other_bound = 91
	elif ((c_code in range(65, 91))and (sc_code > 90)):
		bound = 90
		other_bound = 64
	elif ((c_code in range(97, 123))and (sc_code < 97)):
		bound = 97
		other_bound = 123
	else:
		bound = 122
		other_bound = 96 

	#print c_code, sc_code, bound, other_bound
	#print other_bound + (sc_code - bound)

	return chr(other_bound + (sc_code - bound ))

def caesar(message, shift):

	if (shift < 0) :
		shift = -1 * (abs(shift)%26)
	else :
		shift = shift%26

	#print (shift)
	encrypted = ''
	for char in message:
		if (in_range(char)):
			shifted_char = chr(ord(char) + shift)
			#print shifted_char

			if(letter_case(char) != letter_case(shifted_char)):
				shifted_char = cycle_through(char, shifted_char)

		else:
			shifted_char = char

		encrypted += shifted_char
		#print (encrypted)

	return encrypted


def decrypt(message, shift): 
	return caesar(message, (-1)*shift)


#print(caesar('AbCd', -3))
#print(caesar('ZyXw', -100))
#print(caesar('ZyXw', 100))
#print (caesar('ZyXw', 22))
#print(decrypt('VuTs', 22))
#print(caesar('I am 100% awesome', -3))

assert caesar('AbCd', -3) == 'XyZa', "Incorrect encryption"
assert caesar('ZyXw', -100) == 'DcBa', "Incorrect encryption"
assert caesar('ZyXw', 100) == 'VuTs', "Incorrect encryption"
assert caesar('ZyXw', 22) == 'VuTs', "Incorrect encryption"
assert caesar('I am 100% awesome', -3) == 'F xj 100% xtbpljb', "Incorrect encryption"
assert decrypt('VuTs', 22) == 'ZyXw', "Incorrect encryption"
