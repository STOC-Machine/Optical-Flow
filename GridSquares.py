import numpy as np
import cv2
import sys
import glob
from operator import attrgetter
import math

def dot(a,b):#a.b
	return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def vecsub(a,b):#a-b
	return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]
def vecadd(a,b): #a+b
	return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]
def distance(a): #magnitude a
	return math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
def scalarmult(a,b): #Multiply vector a by number b
	return([a[0]*b,a[1]*b,a[2]*b])
def sign(a,b): #Are a and b in the same direction or opposite?
	return(dot(a,b)/abs(dot(a,b)))
def vectordiv(a,b):
	return((a[0]/b[0]+a[1]/b[1]+a[2]/b[2])/3)
def cross(a,b): #a cross b
	return([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
def cross2D(a, b): #a cross b
	return(a[0]*b[1]-a[1]*b[0])
def proj(a,b): #Projection of b onto unit vector a
	return(scalarmult(a,dot(a,b)/(distance(a)*distance(a))))
def denumpify(a): #This is terrible long term. But it works for now!
	return [[a[0][0][0],a[0][0][1]],[a[1][0][0],a[1][0][1]],[a[2][0][0],a[2][0][1]],[a[3][0][0],a[3][0][1]]]

def computeFrameSquares(image):
	red,green,blue=cv2.split(image) #split the image into components.
	testgray=np.minimum(blue,red) #Create a new image with the minimum of b and r channels
	testgray=np.minimum(testgray,green) #Create a new image with minimum of all three channels
	out,ret=cv2.threshold(testgray,120,255,cv2.THRESH_BINARY) #Run a threshold to find only white lines. Interestingly, ret is the image here.
	#try:
	#cv2.imshow('Threshold',ret) #Display the thresholded image
	#except:
	#exit=1 #If that's not working, your screwed. Just give up now.
	#pass
	dump,contours,hierarchy=cv2.findContours(ret,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) #Get contours in image in a list.
	contours.sort(key=cv2.contourArea,reverse=True) #Sort them by area. Trust me. Saves time because there are a ton of contours with 0 area.
	contour=0 #Iterator
	squares=[] #List of squares to add to.
	while(contour<len(contours) and cv2.contourArea(contours[contour])>1000): #Loop until area is too small or all are done
		newsquare=gridSquare(contours[contour])
		#print cv2.contourArea(contours[contour])
		epsilon = 0.01*cv2.arcLength(newsquare.contour,True) #Set up for simplifying contours
		newsquare.corners=cv2.approxPolyDP(newsquare.contour,epsilon,True) #Actually simplifying
		if(len(newsquare.corners)==4): #If the simplified version has 4 sides
			squares.append(newsquare) #And mark it as a square
		contour+=1 #Iterate
	return squares

class gridSquare:
	camRot=None #Rotation Vector of Camera
	normal=[] #Normal vector pointing out of the square	
	rvec=None
	camerapos=None
	score=0 #How good is this square. See compare square normals for more
	corners=[] #Image coordinates of square corners
	contour=None #Unsimplified image coordinates of square corners
	location=[] #Square location in camera coordinates
	def getPosStats(self,CameraMatrix,distortionCoefficients,objectpoints):
		camvalues=[]
		#print self.corners
		tempcorners=self.corners.reshape(4,2,1).astype(float) #The points need to be floats, and in a specific shape
		#print tempcorners,objectpoints
		#print objectpoints.shape,tempcorners.shape
		inliers,self.rvec,tvec=cv2.solvePnP(objectpoints,tempcorners,CameraMatrix,distortionCoefficients) #Where the magic happens. Turns gets vector from camera to center of square
		rotMatrix=cv2.Rodrigues(self.rvec)
		camerapos=np.multiply(cv2.transpose(rotMatrix[0]), -1).dot(tvec)
		camToGridTransform=np.concatenate((cv2.transpose(rotMatrix[0]),camerapos),axis=1)
		gridToCamTransform=np.linalg.inv(np.concatenate((camToGridTransform,np.array([[0,0,0,1]])),axis=0))
		self.camRot=list(camToGridTransform.dot(np.array([0,0,1,0])))
		self.normal=gridToCamTransform.dot(np.array([0,0,1,0]))
		#print self.normal
		self.camerapos=camerapos
		self.location=[camerapos[0][0],camerapos[1][0],camerapos[2][0]]

	def alignSquares(self,guess,CameraMatrix,distortionCoefficients,objectpoints):
		alignmentvals=[0,0,0,0]
		alignmentvals[0]=dot(self.camRot,guess)
		for rot in range(1,4):
			tempcorners=np.roll(self.corners,rot,axis=0)
			tempcorners=tempcorners.reshape(4,2,1).astype(float)
			inliers,self.rvec,tvec=cv2.solvePnP(objectpoints,tempcorners,CameraMatrix,distortionCoefficients) #Where the magic happens. Turns gets vector from camera to center of square
			rotMatrix=cv2.Rodrigues(self.rvec)
			camerapos=np.multiply(cv2.transpose(rotMatrix[0]), -1).dot(tvec)
			camToGridTransform=np.concatenate((cv2.transpose(rotMatrix[0]),camerapos),axis=1)
			gridToCamTransform=np.linalg.inv(np.concatenate((camToGridTransform,np.array([[0,0,0,1]])),axis=0))
			alignmentvals[rot]=dot(list(camToGridTransform.dot(np.array([0,0,1,0]))),guess)
		#print alignmentvals,alignmentvals.index(max(alignmentvals))
		#print self.corners
		#print self.location
		self.corners=np.roll(self.corners,alignmentvals.index(max(alignmentvals)),axis=0)
		#print self.corners
		self.getPosStats(CameraMatrix,distortionCoefficients,objectpoints)

	def compareSquareNormals(self,square,img):
		tempcross=cross(self.normal,square.normal)
		edge=0
		for point in square.corners:
			if(point[0][0]<1 or point[0][0]>len(img[0])-2 or point[0][1]<1 or point[0][1]>len(img)-2):
				edge=1
		if(not edge):
			#score+=abs(dot(cross1,cross2))
			self.score+=1-abs(distance(tempcross)/(distance(square.normal)*distance(self.normal)))
	def __init__(self,contour):
		self.contour=contour
		

CameraMatrix=np.array([[811.75165344, 0., 317.03949866],[0., 811.51686214, 247.65442989],[0., 0., 1.]]) #Values found in CalibrationValues.txt
distortionCoefficients=np.array([-3.00959078e-02, -2.22274786e-01, -5.31335928e-04, -3.74777371e-04, 1.80515550e+00]) #Values found in Calibration Values.txt
distortionCoefficients=distortionCoefficients.reshape(5,1) #Needs to be this shape

	
font = cv2.FONT_HERSHEY_SIMPLEX #Used for drawing text.
camera=0 #Will be used to test camera loading
if(len(sys.argv)<2): #If no arguments passed
	camera=cv2.VideoCapture(1) #Load the webcam
	filenames=[] #Don't give any filenames
else:
	filenames=glob.glob(sys.argv[1]) #Get the filenames from the command
	print(filenames) #Print them, 'cause why not?
exit=0 #Don't stop running yet

BestCamRotGuess=0

def getSquareStats(image,CameraMatrix,distortionCoefficients,BestCamRotGuess=0):
	squarelength=28.5 #Needs to be a float, in cm of real length of the squares
	squareGap=2.5
	objectpoints=np.array([[[-squarelength/2,-squarelength/2,0]],[[-squarelength/2,squarelength/2,0]],[[squarelength/2,squarelength/2,0]],[[squarelength/2,-squarelength/2,0]]],np.float32) #3d grid square coordinates
	objectpoints=objectpoints.reshape(4,3,1) #Needs to be this shape
	squares=computeFrameSquares(image)
	for square in squares:
		square.getPosStats(CameraMatrix,distortionCoefficients,objectpoints)
	index1=0
	while(index1<len(squares)):
		index2=0
		while(index2<len(squares)):
			squares[index1].compareSquareNormals(squares[index2],image)
			index2+=1
		index1+=1
	squares.sort(key=attrgetter('score'),reverse=True)
	if(len(squares)>0):
		scorethreshold=max(.95*squares[0].score,1.0)
		squares=list(filter(lambda x:x.score>scorethreshold,squares))
	for square in squares:
		if(BestCamRotGuess==0):
			BestCamRotGuess=squares[0].camRot
		square.alignSquares(BestCamRotGuess,CameraMatrix,distortionCoefficients,objectpoints)
	print(squares)
	return squares,BestCamRotGuess
"""
while(len(filenames)>0 or not exit): #If there are more files, or we haven't quit yet
	if(len(filenames)>0): #If we're running purely on files
		exit=1 #Make it quit when we're done with files
		filename=filenames.pop(0) #And get the first file in the list
		try: 
			img=cv2.imread(filename) #Read the image
		except:
			continue #Unless you can't. Then skip it.
	elif(camera): #If using webcam
		ret, img = camera.read() #Read from webcam
	else: #If things are weird, just quit
		break
	if(img==None): #Do make sure that there's an image
		break
	outimg=np.copy(img) #Copy the image. Not really needed, but can be nice long term
	#print img.shape
	birdsview=np.zeros([1000,1000,3],dtype=np.uint8)
	cv2.circle(birdsview,(int(birdsview.shape[0]/2),int(birdsview.shape[1]/2)),5,(255,255,0),-1)

	squares,BestCamRotGuess=getSquareStats(img,CameraMatrix,distortionCoefficients,BestCamRotGuess) #It's a magic function! Yay!
	#print(contour,len(squares)) #Print the # of squares found
	#print(len(img),len(img[0]))

	gluedSquareCorners=[]
	gluedSquareCoords=[]
	squarelength=28.5
	squareGap=2
	baseobjectpoints=[[-squarelength/2,-squarelength/2,0],[-squarelength/2,squarelength/2,0],[squarelength/2,squarelength/2,0],[squarelength/2,-squarelength/2,0]]
	if(BestCamRotGuess!=0):
		tempcamline=vecadd(scalarmult(BestCamRotGuess,50),[birdsview.shape[0]/2,birdsview.shape[0]/2,0])
		cv2.line(birdsview,(int(tempcamline[0]),int(tempcamline[1])),(int(birdsview.shape[0]/2),int(birdsview.shape[1]/2)),(0,0,255),1,cv2.LINE_AA)
	for square in squares:
		tempvec=vecsub(square.location,squares[0].location)
		INeedAnIndex=0
		while(INeedAnIndex<4):
			tempdrawvec=vecadd(square.location,baseobjectpoints[INeedAnIndex])
			tempdrawvec2=vecadd(square.location,baseobjectpoints[INeedAnIndex-1])
			INeedAnIndex+=1
			cv2.line(birdsview,(int(tempdrawvec[0]+birdsview.shape[0]/2),int(tempdrawvec[1]+birdsview.shape[1]/2)),(int(tempdrawvec2[0]+birdsview.shape[0]/2),int(tempdrawvec2[1]+birdsview.shape[1]/2)),(255,255,255),3,cv2.LINE_AA)
		tempvec[0]=(squarelength+squareGap)*round(tempvec[0]/(squarelength+squareGap),0)
		tempvec[1]=(squarelength+squareGap)*round(tempvec[1]/(squarelength+squareGap),0)
		tempvec[2]=0
		for i in baseobjectpoints:
			gluedSquareCoords.append([[vecadd(i,tempvec)[0]],[vecadd(i,tempvec)[1]],[vecadd(i,tempvec)[2]]])
		for i in denumpify(square.corners):
			gluedSquareCorners.append([[i[0]],[i[1]]])
		#print tempvec
		#print(square.location)
		x=0
		y=0
		for point in square.corners:
			x+=point[0][0]
			y+=point[0][1]
		x=int(x/4)
		y=int(y/4)
		cv2.putText(img,str(int(abs(square.location[2])))+" "+str(int(square.score*100)),(x,y), font, 1,(255,255,255),1,cv2.LINE_AA)

		cv2.polylines(img,[square.corners],True,(255,0,0)) #Draw both squares
		#print square.location
		#print square.side1
		cv2.drawContours(img,square.contour,True,(0,255,0))
		
		#print(len([square for square in squares if square.score > scorethreshold]))
	if(len(squares)>0):
		gluedSquareCorners=np.asarray(gluedSquareCorners).astype(float)
		gluedSquareCoords=np.asarray(gluedSquareCoords).astype(float)
		gluedSquareCorners.reshape(len(gluedSquareCorners),2,1)
		gluedSquareCoords.reshape(len(gluedSquareCoords),3,1)
		for square2 in squares:
			print square2.corners
		for square2 in squares:
			print vecsub(square2.location,squares[0].location)
		print gluedSquareCorners
		print gluedSquareCoords
		inliers,fullrvec,fulltvec=cv2.solvePnP(gluedSquareCoords,gluedSquareCorners,CameraMatrix,distortionCoefficients) #Where the magic happens. Turns gets vector from camera to center of square
		print fulltvec,fullrvec
		rotMatrix=cv2.Rodrigues(fullrvec)
		camerapos=np.multiply(cv2.transpose(rotMatrix[0]), -1).dot(fulltvec)
		print camerapos
		print squares[0].location
		camToGridTransform=np.concatenate((cv2.transpose(rotMatrix[0]),camerapos),axis=1)
		gridToCamTransform=np.linalg.inv(np.concatenate((camToGridTransform,np.array([[0,0,0,1]])),axis=0))
		camRot=list(camToGridTransform.dot(np.array([0,0,1,0])))
		tempcamline2=vecadd(scalarmult(camRot,50),[birdsview.shape[0]/2,birdsview.shape[0]/2,0])
		cv2.line(birdsview,(int(tempcamline2[0]),int(tempcamline2[1])),(int(birdsview.shape[0]/2),int(birdsview.shape[1]/2)),(255,0,255),1,cv2.LINE_AA)
	print("") #Divider line
	try:
		cv2.imshow("hi",img) #This is mainly to let my borked python3 install, which can't display images, work.
		cv2.imshow("Birds eye view",birdsview)
		if(camera): #If we're doing video
			if(len(squares)>0):
				BestCamRotGuess=squares[0].camRot
			if cv2.waitKey(1) & 0xFF == ord('q'): #Let the program run while waiting for q to be pressed
				break #Exit
		else: #If it's just files,
			BestCamRotGuess=0
			cv2.waitKey(0) #Wait for a key to continue on
			cv2.destroyAllWindows() #And remove the old window
	except:
		pass
#cv2.destroyAllWindows() #And don't leave silly windows behind.
"""