import bcrypt

passwd = b's$cret12'

salt = bcrypt.gensalt()
hashed = bcrypt.hashpw(passwd, salt)

print(hashed)
if bcrypt.checkpw(passwd, hashed):
    print("match")
else:
    print("does not match")