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

def computeFrameSquares(image):
	red,green,blue=cv2.split(image) #split the image into components.
	testgray=np.minimum(blue,red) #Create a new image with the minimum of b and r channels
	testgray=np.minimum(testgray,green) #Create a new image with minimum of all three channels
	out,ret=cv2.threshold(testgray,120,255,cv2.THRESH_BINARY) #Run a threshold to find only white lines. Interestingly, ret is the image here.
	"""
	try:
		cv2.imshow('Threshold',ret) #Display the thresholded image
	except:
		#exit=1 #If that's not working, your screwed. Just give up now.
		pass
	"""
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
	camvec=[] #Vector pointing to camera, in camera coords
	side1=[] #Unit vector from center to side1 of square, in camera coords
	side2=[] #Unit vector from center to side1 of square, in camera coords
	normal=[] #Normal vector pointing out of the square	

	score=0 #How good is this square. See compare square normals for more
	corners=[] #Image coordinates of square corners
	contour=None #Unsimplified image coordinates of square corners
	location=[] #Square location in camera coordinates
	def getPosStats(self):
		tempcorners=self.corners.reshape(4,2,1).astype(float) #The points need to be floats, and in a specific shape
		inliers,rvec,tvec=cv2.solvePnP(objectpoints,tempcorners,CameraMatrix,distortionCoefficients) #Where the magic happens. Turns gets vector from camera to center of square
		inliers,rvec2,tvec2=cv2.solvePnP(secondObjectCorners,tempcorners,CameraMatrix,distortionCoefficients)
		inliers,rvec3,tvec3=cv2.solvePnP(thirdObjectCorners,tempcorners,CameraMatrix,distortionCoefficients)
		#print(secondObjectCorners)
		#print rvec
		#print rvec2
		#print rvec3

		#print(distance(vecsub(tvec2,tvec)),distance(vecsub(tvec3,tvec)))
		line1=scalarmult(vecsub(tvec2,tvec),1/distance(vecsub(tvec2,tvec)))
		line2=scalarmult(vecsub(tvec3,tvec),1/distance(vecsub(tvec3,tvec)))
		self.camvec=[float(tvec[0]),float(tvec[1]),float(tvec[2])]
		self.side1=[float(line1[0]),float(line1[1]),float(line1[2])]
		self.side2=[float(line2[0]),float(line2[1]),float(line2[2])]
		self.normal=cross(self.side1,self.side2)
		tempx=proj(self.side1,self.camvec)
		tempy=proj(self.side2,self.camvec)
		tempz=proj(self.normal,self.camvec)
		self.location=[sign(tempx,self.side1)*distance(tempx),sign(tempy,self.side2)*distance(tempy),sign(tempz,self.normal)*distance(tempz)]
		#self.location[0]=sign(temp,self.side1)*distance(temp)

#		self.location[1]=sign(temp,self.side2)*distance(temp)
#		self.location[2]=sign(temp,self.normal)*distance(temp)
		#print self.location
		#print ""
	def compareSquareNormals(self,square,img):
		tempcross=cross(self.normal,square.normal)
		edge=0
		for point in square.corners:
			if(point[0][0]<1 or point[0][0]>len(img)-2 or point[0][1]<1 or point[0][1]>len(img)-2):
				edge=1
		if(not edge):
			#score+=abs(dot(cross1,cross2))
			self.score+=1-abs(distance(tempcross)/(distance(square.normal)*distance(self.normal)))
	def getHeight(self):
		height = height=abs(self.location[2])
		return height
	def __init__(self,contour):
		self.contour=contour

squarelength=29.0 #Needs to be a float, in cm of real length of the squares
objectpoints=np.array([[[-squarelength/2,-squarelength/2,0]],[[-squarelength/2,squarelength/2,0]],[[squarelength/2,squarelength/2,0]],[[squarelength/2,-squarelength/2,0]]],np.float32) #3d grid square coordinates
secondObjectCorners=np.array([[[-squarelength/2,0,0]],[[-squarelength/2,squarelength,0]],[[squarelength/2,squarelength,0]],[[squarelength/2,0,0]]],np.float32)
thirdObjectCorners=np.array([[[0,-squarelength/2,0]],[[0,squarelength/2,0]],[[squarelength,squarelength/2,0]],[[squarelength,-squarelength/2,0]]],np.float32)

objectpoints=objectpoints.reshape(4,3,1) #Needs to be this shape
secondObjectCorners=secondObjectCorners.reshape(4,3,1)
thirdObjectCorners=thirdObjectCorners.reshape(4,3,1)
CameraMatrix=np.array([[811.75165344, 0., 317.03949866],[0., 811.51686214, 247.65442989],[0., 0., 1.]]) #Values found in CalibrationValues.txt
distortionCoefficients=np.array([-3.00959078e-02, -2.22274786e-01, -5.31335928e-04, -3.74777371e-04, 1.80515550e+00]) #Values found in Calibration Values.txt

distortionCoefficients=distortionCoefficients.reshape(5,1) #Needs to be this shape

heights=[[0,0],[0,0],[0,0]]
font = cv2.FONT_HERSHEY_SIMPLEX #Used for drawing text.
camera=0 #Will be used to test camera loading
"""
if(len(sys.argv)<2): #If no arguments passed
	camera=cv2.VideoCapture(1) #Load the webcam
	filenames=[] #Don't give any filenames
else:
	filenames=glob.glob(sys.argv[1]) #Get the filenames from the command
	print(filenames) #Print them, 'cause why not?
exit=0 #Don't stop running yet
"""
"""
while(len(filenames)>0 or not exit): #If there are more files, or we haven't quit yet
	heights[2]=heights[1]
	heights[1]=heights[0]
	heights[0]=[0,0]
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
	if(img is None): #Do make sure that there's an image
		break
	outimg=np.copy(img) #Copy the image. Not really needed, but can be nice long term
	squares=computeFrameSquares(img)
	#print(contour,len(squares)) #Print the # of squares found
	#print(len(img),len(img[0]))
	for square in squares:
		square.getPosStats()
	index1=0 #Iterator1
	#print("")

	while(index1<len(squares)): #Loop through squares
		index2=0 #Iterator2 starts where 1 hasn't reached
		score=0
		while(index2<len(squares)): #And loops through squares
			squares[index1].compareSquareNormals(squares[index2])
			index2+=1 #Iterate
		index1+=1 #Iterate
	squares.sort(key=attrgetter('score'),reverse=True)
	if(len(squares)>0):
		averageheight=0
		scorethreshold=.95*squares[0].score
		tvecindex=0
		for square in [square for square in squares if square.score > scorethreshold]:

			#print(square.location)
			height=abs(square.location[2])
			#height=abs(dot(square.camvec,square.normal)/distance(square.normal))
			averageheight+=height
			x=0
			y=0
			for point in square.corners:
				x+=point[0][0]
				y+=point[0][1]
			x=int(x/4)
			y=int(y/4)
			cv2.putText(img,str(int(height))+" "+str(int(square.score*100)),(x,y), font, 1,(255,255,255),1,cv2.LINE_AA)
	
			cv2.polylines(img,[square.corners],True,(255,0,0)) #Draw both squares
			tvecindex+=1
		#print(len([square for square in squares if square.score > scorethreshold]))

		if(tvecindex!=0):
			averageheight=averageheight/tvecindex
			heights[0]=[averageheight,scorethreshold/.95]
			#cv2.putText(img,str(int(averageheight)),(30,30), font, 1,(255,255,255),1,cv2.LINE_AA)
		else:
			pass
			#cv2.putText(img,"N/A",(30,30), font, 1,(255,255,255),1,cv2.LINE_AA)
	else:
		cv2.putText(img,"N/A",(30,30), font, 1,(255,255,255),1,cv2.LINE_AA)
	print("") #Divider line
	weights=[.5,.3,.2]
	totalscore=weights[0]*heights[0][1]+weights[1]*heights[1][1]+weights[2]*heights[2][1]
	if(totalscore!=0):
		heights[0]=[(weights[0]*heights[0][1]*heights[0][0]+weights[1]*heights[1][1]*heights[1][0]+weights[2]*heights[2][1]*heights[2][0])/totalscore,totalscore]
		#cv2.putText(img,str(int(heights[0][0])),(60,60), font, 1,(255,255,255),1,cv2.LINE_AA)
	try:
		cv2.imshow("hi",img) #This is mainly to let my borked python3 install, which can't display images, work.
		if(camera): #If we're doing video
			if cv2.waitKey(1) & 0xFF == ord('q'): #Let the program run while waiting for q to be pressed
				break #Exit
		else: #If it's just files,
			cv2.waitKey(0) #Wait for a key to continue on
			cv2.destroyAllWindows() #And remove the old window
	except:
		pass
#cv2.destroyAllWindows() #And don't leave silly windows behind.
"""