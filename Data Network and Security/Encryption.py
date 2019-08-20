#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 16:34:05 2018

@author: stevencocke
"""
##Created using Python3.x!!!
# =============================================================================
# Exercise 1
# =============================================================================
##stores the word and key from user input
word = input("Enter a phrase to encrypt: ")
key = input("Enter an integer key between 1 and 25: ")
##sets a blank encrypted string. Then goes character by character of the input
##string and takes the ord + key of that character. Checks to see if the
##character is alphabetical and only takes it if it is, assigning it to the
##encrypted variable. Handles both uppercase and lowercase letters as shown
##in the if and elif statements. 
encrypted = ''
for c in word:
    c_cipher = ord(c) + int(key)
    if c.isalpha()==False:
        encrypted = encrypted + c
    elif c.islower():
        if c_cipher > 122:
            encrypted += chr( c_cipher%122 + 96)
        else:
            encrypted += chr( c_cipher )
    else:
        if c_cipher > 90:
            encrypted += chr(c_cipher%90 + 64)
        else:
            encrypted += chr( c_cipher )
   
##prints the encrypted message
print("Your encrypted message: ", encrypted)
print()

# =============================================================================
# Exercise 2         
# =============================================================================
##Create a word input and then make it lowercase
word2 = input("Exercise 2: enter encrypted message: ")
word2 = word2.lower()
##Create empty dictionaries to reference later
chr_count={}
chr_freq={}
##if the character is alphabetical, add it to the dictionary with a +1 to the
## count if it is already there. If it is not alphabetical then pass it.
for i in word2:
    if i.isalpha():
        if i in chr_count:
            chr_count[i] += 1
        else:
            chr_count[i]=1
    else:
        pass
##find the counts divided by the total length of the input (only the alphabeti
##cal values)
for i in chr_count:
    chr_freq[i] = chr_count[i] / sum(chr_count.values())
    
##create function encrypt to encrypt messages
def encrypt(char, key):
    char_ord=ord(char) + key
    if char_ord > 122:
        return (char_ord%122) + 96
    else:
        return char_ord
##create function decrypt to decrypt messages. Add 26 when there is a negativ
##e number so that the alphabet wraps around
def decrypt(char, key):
    char_ord = ord(char) - key
    if char.isalpha():
        if char_ord < 97:
            return (char_ord) + 26
        else:
            return char_ord
    else:
        return ord(char)
## creates a dictionary alphabet_freq. This was taken from Wikipedia and 
##represents the English model probabilities of each letter in the alphabet
alphabet_freq={
        'a':0.08167,
        'b':0.01492,
        'c':0.02782,
        'd':0.04253,
        'e':0.12702,
        'f':0.02228,
        'g':0.02015,
        'h':0.06094,
        'i':0.06966,
        'j':0.00153,
        'k':0.00772,
        'l':0.04025,
        'm':0.02406,
        'n':0.06749,
        'o':0.07507,
        'p':0.01929,
        'q':0.00095,
        'r':0.05987,
        's':0.06327,
        't':0.09056,
        'u':0.02758,
        'v':0.00978,
        'w':0.02360,
        'x':0.00150,
        'y':0.01974,
        'z':0.00074}
##creates empty dictionary key_total to be referenced later
key_total = {}
##this loops for an index from 0 to 25 (a=0) and for each iteration, places
##frequency we found for a particular letter in the input string and
##multiplies it with the frequency in the alphabet_freq dictionary,
##depending on the shift, ofcourse. For each iteration, it is added to a 
##temperory (temp) dictionary and then these products are summed up and placed
##in the dictionary key_total for later use. The temp dictionary is then 
##emptied at the start of a new iteration
for index in range(26):
    temp=[]
    for character in chr_freq:
        temp.append(alphabet_freq[chr(encrypt(character, index))]* chr_freq[character])
    key_total[index] = sum(temp)
###This prints out the key (iteration), the summation that was stored in
##key_total (the correlation resulting from the product discussed previously)
##and the decrypted message, which is a function that was defined earlier
print()
for i in key_total:
    out_string = ''
    for char in word2:
        out_string += chr(decrypt(char, i))
    print(i, key_total[i], out_string)
## =============================================================================
## Exercise 3        
## =============================================================================
###This code was retrieved from:
###https://www.example-code.com/python/aes_chacha20_binary_data.asp
###the jpg file location was changed to generic before submitting
#
#import sys
#import chilkat
#
##  This example requires the Chilkat Crypt API to have been previously unlocked.
##  See Unlock Chilkat Crypt for sample code.
#
##  Load a small JPG file to be encrypted/decrypted.
#jpgBytes = chilkat.CkBinData()
#success = jpgBytes.LoadFile("file_location/starfish.jpg")
#if (success != True):
#    print("Failed to load JPG file.")
#    sys.exit()
#
##  Show the unencrypted JPG bytes in Base64 format.
##  (The "base64_mime" encoding was added in Chilkat v9.5.0.67.
##  The "base64" encoding emits a single line of base64, whereas
##  "base64_mime" will emit multi-line base64 as it would appear
##  in MIME.)
#print(jpgBytes.getEncoded("base64_mime"))
#
##  Sample base64_mime JPG data:
#
##  	/9j/4AAQSkZJRgABAgEASABIAAD/7Q18UGhvdG9zaG9wIDMuMAA4QklNA+0AAAAAABAASAAAAAEA
##  	AQBIAAAAAQABOEJJTQPzAAAAAAAIAAAAAAAAAAE4QklNBAoAAAAAAAEAADhCSU0nEAAAAAAACgAB
##  	AAAAAAAAAAI4QklNA/UAAAAAAEgAL2ZmAAEAbGZmAAYAAAAAAAEAL2ZmAAEAoZmaAAYAAAAAAAEA
##  	MgAAAAEAWgAAAAYAAAAAAAEANQAAAAEALQAAAAYAAAAAAAE4QklNBBQAAAAAAAQAAAABOEJJTQQM
##  	...
#
#crypt = chilkat.CkCrypt2()
#
##  Specify the encryption to be used.
##  First we'll do AES-128 CBC
#crypt.put_CryptAlgorithm("aes")
#crypt.put_CipherMode("cbc")
#crypt.put_KeyLength(128)
#
#ivHex = "000102030405060708090A0B0C0D0E0F"
#crypt.SetEncodedIV(ivHex,"hex")
#
#keyHex = "000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F"
#crypt.SetEncodedKey(keyHex,"hex")
#
##  Do the in-place 128-bit AES CBC encryption.
##  The contents of jpgBytes are replaced with the encrypted bytes.
#success = crypt.EncryptBd(jpgBytes)
#if (success != True):
#    print(crypt.lastErrorText())
#    sys.exit()
#
##  Examine the JPG bytes again.  The bytes should be different because they are encrypted:
#print(jpgBytes.getEncoded("base64_mime"))
#
##  Sample base64_mime encrypted JPG data:
#
##  	sbz0babt1WCkQf5xKMdg/baZAcUBO5GVUUDF2BjVqmd+HrqKN+t6hAcqakL/bdo0q9hYmow0Tp1e
##  	AQ9V9DOiifQUZqWVkR+kL/c45bq8JGFDvgNl0djPt+yYhV789IB/fPH0upx+/ad++WNOlv1IxGMr
##  	Y1x1oERU/IsiEzafUJdI4kZ6FQo2IPGMF/Rm1h79I7hP1yYUFxvJyz+PzaySAUH1nLsNHyDVY5VY
##  	O90aH3steRSYbz8C8UF9wQ3qqEIXQNnnixvoNDnmHyY39VoVBI5F6rnPwYDfAk2t8tmuryFqvwAu
##  	...
###ECB and CBC
###This code was received from:
###https://gist.github.com/Shathra/92bc0babadcf19cec36fb25fe4bdecb2
##  Decrypt to restore back to the original:
#success = crypt.DecryptBd(jpgBytes)
#if (success != True):
#    print(crypt.lastErrorText())
#    sys.exit()
#
#print(jpgBytes.getEncoded("base64_mime"))
#
##  	/9j/4AAQSkZJRgABAgEASABIAAD/7Q18UGhvdG9zaG9wIDMuMAA4QklNA+0AAAAAABAASAAAAAEA
##  	AQBIAAAAAQABOEJJTQPzAAAAAAAIAAAAAAAAAAE4QklNBAoAAAAAAAEAADhCSU0nEAAAAAAACgAB
##  	AAAAAAAAAAI4QklNA/UAAAAAAEgAL2ZmAAEAbGZmAAYAAAAAAAEAL2ZmAAEAoZmaAAYAAAAAAAEA
##  	MgAAAAEAWgAAAAYAAAAAAAEANQAAAAEALQAAAAYAAAAAAAE4QklNBBQAAAAAAAQAAAABOEJJTQQM
##  	...
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as img
#from Crypto.Cipher import AES
#from Crypto import Random
#
##Get the image
#kyle = img.imread('kyle.png')
#
##Before encryption
#imgplot=plt.imshow(kyle)
#plt.figure()
#
##Conversion to string
#kyle = np.asarray(kyle * 256, dtype=int)
#msg = kyle.tostring()
#
##Encryption
#key = "keymust16bytelong"
#cipher_ecb = AES.new( key)
#
#iv = Random.new().read(AES.block_size)
#cipher_cbc = AES.new( key, AES.MODE_CBC, iv)
#cipher = AES.new(key)
#msg_ecb = cipher_ecb.encrypt( msg)
#msg_cbc = cipher_cbc.encrypt( msg)
#
##Showing message encrypted in ECB mode
#kyle = np.fromstring(msg_ecb,dtype=int)
#kyle = np.asarray(kyle, dtype=np.float32)
#kyle = kyle / 256
#kyle = kyle.reshape(300,250,4)
#imgplot=plt.imshow(kyle)
#plt.figure()
#
##Showing message encrypted in CBC mode
#kyle = np.fromstring(msg_cbc,dtype=int)
#kyle = np.asarray(kyle, dtype=np.float32)
#kyle = kyle / 256
#kyle = kyle.reshape(300,250,4)
#imgplot=plt.imshow(kyle)
#plt.figure()
#
##Showing decrypted image
#kyle = np.fromstring( cipher_cbc.decrypt( msg_cbc),dtype=int)
#kyle = np.asarray(kyle, dtype=np.float32)
#kyle = kyle / 256
#kyle = kyle.reshape(300,250,4)
#imgplot=plt.imshow(kyle)
#plt.show()