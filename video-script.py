import cv2

glasses = cv2.imread("./glasses.png", -1)
moustache = cv2.imread("./mustache.png", -1)

capture = cv2.VideoCapture(0)

eyes_cascade = cv2.CascadeClassifier("./frontalEyes35x16.xml")
nose_cascade = cv2.CascadeClassifier("./Nose18x15.xml")

while True:

	ret, frame = capture.read()

	if ret == False:
		continue

	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA) 

	eyes = eyes_cascade.detectMultiScale(frame, 1.3, 5)
	noses = nose_cascade.detectMultiScale(frame, 1.3, 5)

	for (x, y, w, h) in eyes:
		# cv2.rectangle(frame, (x, y) , (x+w, y+h), (255, 0, 0), 2)

		# to remove index out of bound
		glasses = cv2.resize(glasses, (w, h)) 

		gw, gh, gc = glasses.shape
		for i in range(0, gw):
			for j in range(0, gh):
				if glasses[i, j][3] != 0:
					frame[y + i, x + j] = glasses[i, j]

	for (x, y, w, h) in noses:
		# cv2.rectangle(frame, (x, y) , (x+w, y+h), (255, 233, 0), 2)

		# to remove index out of bound
		moustache = cv2.resize(moustache, (w, h))

		mw, mh, mc = moustache.shape
		for i in range(0, mw):
			for j in range(0, mh):
				if moustache[i, j][3] != 0:
					try:
						frame[y + i + int(h / 2.0), x + j] = moustache[i, j]
					except:
						pass

	cv2.imshow("WoW", frame)

	key_pressd = cv2.waitKey(1) & 0xFF
	if key_pressd == ord('q'):
		break


capture.release()
cv2.destroyAllWindows()