import numpy as np
import cv2
import sys
import glob
from operator import attrgetter
import math

#Vector functions. Probably should be replaced with numpy versions
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

def denumpify(a): #Numpy data types were annoying me. This should be removed eventually.
	return [[a[0][0][0],a[0][0][1]],[a[1][0][0],a[1][0][1]],[a[2][0][0],a[2][0][1]],[a[3][0][0],a[3][0][1]]]

# Compute Frame Squares:
# Takes an image, and returns a list of all "squares" in the image
#
def computeFrameSquares(img):
	red,green,blue=cv2.split(img) #split the image into components.
	testgray=np.minimum(blue,red) #Create a new image with the minimum of b and r channels
	testgray=np.minimum(testgray,green) #Create a new image with minimum of all three channels
	#out,ret=cv2.threshold(testgray,80,255,cv2.THRESH_BINARY) #Run a threshold to find only white lines. Interestingly, ret is the image here.
	out,ret=cv2.threshold(testgray,120,255,cv2.THRESH_BINARY) #Logitech camera threshold
	try:
		cv2.imshow('Threshold',ret) #Display the thresholded image
	except:
		pass
	dump,contours,hierarchy=cv2.findContours(ret,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) #Get contours in image in a list.
	contours.sort(key=cv2.contourArea,reverse=True) #Sort them by area. Trust me. Saves time because there are a ton of contours with 0 area.
	contour=0 #Iterator
	squares=[] #List of squares to add to.
	while(contour<len(contours) and cv2.contourArea(contours[contour])>1000): #Loop until area is too small or all are done
		newsquare=gridSquare(contours[contour])
		epsilon = 0.01*cv2.arcLength(newsquare.contour,True) #Set up for simplifying contours
		newsquare.corners=cv2.approxPolyDP(newsquare.contour,epsilon,True) #Actually simplifying
		if(len(newsquare.corners)==4): #If the simplified version has 4 sides
			squares.append(newsquare) #Mark it as a square
		contour+=1 #Iterate
	return squares

class gridSquare:
	camRot=None #Rotation vector of camera aligned to grid square axes.
	normal=[] #Normal vector pointing out of the square	
	rvec=None #rvec returned by solvePNP
	camerapos=None #Equivalent to location. Used for internal testing reasons
	tvec=None #tvec returned by solvePNP ransac
	score=0 #How good is this square. See compare square normals for more
	corners=[] #Image coordinates of square corners
	contour=None #Unsimplified image coordinates of square corners
	location=[] #Square location in camera coordinates

	# Get Position Stats
	# Takes the square's corners found in the image, corresponding 3d coordinates, and intrinsic camera information
	# Sets fields related to extrinsic camera information: camRot, normal, camerapos, location
	# Note that the square might be bogus until you check its score, and camera extrinsics are unaligned until align squares is called.
	def getPosStats(self,CameraMatrix,distortionCoefficients,objectpoints):
		camvalues=[]
		tempcorners=self.corners.reshape(4,2,1).astype(float) #The points need to be floats, and in a specific shape
		inliers,self.rvec,self.tvec=cv2.solvePnP(objectpoints,tempcorners,CameraMatrix,distortionCoefficients) #Where the magic happens. Turns gets vector from camera to center of square
		rotMatrix=cv2.Rodrigues(self.rvec)
		camerapos=np.multiply(cv2.transpose(rotMatrix[0]), -1).dot(self.tvec)
		camToGridTransform=np.concatenate((cv2.transpose(rotMatrix[0]),camerapos),axis=1)
		gridToCamTransform=np.linalg.inv(np.concatenate((camToGridTransform,np.array([[0,0,0,1]])),axis=0))
		self.camRot=list(camToGridTransform.dot(np.array([0,0,1,0])))
		self.normal=gridToCamTransform.dot(np.array([0,0,1,0]))
		self.camerapos=camerapos
		self.location=[camerapos[0][0],camerapos[1][0],camerapos[2][0]]

	# Align Squares (needed because ABCD has different camera location from BCDA)
	# Takes a squares corners, camera intrisics, object information, and a camera rotation to align to.
	# Finds the camera position that gives a rotation vector (unit vector in cam direction using grid axes) closest to guess.
	def alignSquares(self,guess,CameraMatrix,distortionCoefficients,objectpoints):
		alignmentvals=[0,0,0,0] # Holds scores for different alignments
		alignmentvals[0]=dot(self.camRot,guess) # Since we have some data, use it.
		for rot in range(1,4): # Loops through possible orders, ABCD,BCDA,CDAB,DABC
			tempcorners=np.roll(self.corners,rot,axis=0) #shift to the rot permutation
			tempcorners=tempcorners.reshape(4,2,1).astype(float) #reshape for solvePNP
			inliers,self.rvec,self.tvec=cv2.solvePnP(objectpoints,tempcorners,CameraMatrix,distortionCoefficients) #Where the magic happens. Gets vector from camera to center of square
			rotMatrix=cv2.Rodrigues(self.rvec) # Getting rotation information for change to square center at origin
			camerapos=np.multiply(cv2.transpose(rotMatrix[0]), -1).dot(self.tvec) # Finds camera position
			camToGridTransform=np.concatenate((cv2.transpose(rotMatrix[0]),camerapos),axis=1) # Creates a transformation matrix
			gridToCamTransform=np.linalg.inv(np.concatenate((camToGridTransform,np.array([[0,0,0,1]])),axis=0)) #Inverts it
			alignmentvals[rot]=dot(list(camToGridTransform.dot(np.array([0,0,1,0]))),guess) # Compare camera unit vector to guess
		self.corners=np.roll(self.corners,alignmentvals.index(max(alignmentvals)),axis=0) # Pick the orientation that was best
		self.getPosStats(CameraMatrix,distortionCoefficients,objectpoints) # Recompute camera location for that orientation

	# Compare Square Normal Vectors
	# Increments the score of two squares based on how parallel their normal vectors are.
	# dim is the dimensions of the image
	def compareSquareNormals(self,square,dim):
		tempcross=cross(self.normal,square.normal) #Using 1-cross product due to larger change of sin when parallel
		edge=0
		for point in square.corners:
			if(point[0][0]<1 or point[0][0]>dim[1]-2 or point[0][1]<1 or point[0][1]>dim[0]-2): #Contours on the edge don't improve scores
				edge=1
		if(not edge):
			self.score+=1-abs(distance(tempcross)/(distance(square.normal)*distance(self.normal))) #Increment the score

	# Init: Creates a square object from a given contour
	def __init__(self,contour):
		self.contour=contour

# Get Square Stats
# Returns list of squares with locations, sorted by score, aligned to BestCamRotGuess
# Needs camera intrinsics. If BestCamRotGuess is 0, will pick an arbitrary square to align to.
def getSquareStats(img,CameraMatrix,distortionCoefficients,BestCamRotGuess):
	squarelength=28.5 #Needs to be a float, in cm of real length of the squares
	squareGap=2.5 #Float in cm of width of gaps between squares.
	objectpoints=np.array([[[-squarelength/2,-squarelength/2,0]],[[-squarelength/2,squarelength/2,0]],[[squarelength/2,squarelength/2,0]],[[squarelength/2,-squarelength/2,0]]],np.float32) #3d grid square coordinates
	objectpoints=objectpoints.reshape(4,3,1) #Needs to be this shape

	squares=computeFrameSquares(img) # Find squares in the image
	for square in squares:
		square.getPosStats(CameraMatrix,distortionCoefficients,objectpoints) #Get camera location for those squares

	#Compare each pair of squares
	index1=0
	while(index1<len(squares)):
		index2=0
		while(index2<len(squares)):
			squares[index1].compareSquareNormals(squares[index2],img.shape)
			index2+=1
		index1+=1
	#Sort by score
	squares.sort(key=attrgetter('score'),reverse=True)
	#Filter out low scores if there are squares
	if(len(squares)>0):
		scorethreshold=max(.95*squares[0].score,1.0)
		squares=list(filter(lambda x:x.score>scorethreshold,squares))

	#Align squares.
	for square in squares:
		if(BestCamRotGuess==0): #If we don't have a guess, use the highest score square
			BestCamRotGuess=squares[0].camRot
		square.alignSquares(BestCamRotGuess,CameraMatrix,distortionCoefficients,objectpoints)
	return squares,BestCamRotGuess

