			 ________    ________      ________      ___       __       ___          
			|\   __  \  |\   __  \    |\   __  \    |\  \     |\  \    |\  \         
			\ \  \|\  \ \ \  \|\  \   \ \  \|\  \   \ \  \    \ \  \   \ \  \        
 			 \ \   ____\ \ \   _  _\   \ \  \\\  \   \ \  \  __\ \  \   \ \  \       
 			  \ \  \___|  \ \  \\  \|   \ \  \\\  \   \ \  \|\__\_\  \   \ \  \____  
 			   \ \__\      \ \__\\ _\    \ \_______\   \ \____________\   \ \_______\
			    \|__|       \|__|\|__|    \|_______|    \|____________|    \|_______|

					     AI chatbot for the UWM Archives
                  			  Matthew Cavenaile (and Claude/ChatGPT)                               
                          		        matt.cavenaile@gmail.com                                         
                                                                        
Purpose & Scope
============================================================================================================================
The purposes of this chatbot are to assist patrons with basic to intermediate level reference requests.
Prowl should be able to assist patrons in the following ways:
	- Answer general Archives questions (What are your hours? What kinds of records are available at the archives?)
	- Narrow down archival search queries
	- Identify specific collection finding aids
	- If Archives is the incorrect authority, Prowl should be able to
	  refer patrons to the correct UWM libraries resource

Future implementations
	- Notification system linked to askarch@uwm.edu (help patrons setup appointments)
	- Ability to answer Genealogy/WTMJ reference requests


To-do
============================================================================================================================
+ Possible to setup a notification system via AskArch if a patron would like to look at a collection? Clickable links in chat?
+ Future functionality - integrate w/ WTMJ collection?
+ Add text-bubbls to know when the bot is thinking



Limitations
============================================================================================================================
+ It is worth noting that Prowl uses an older model of ChatGPT (3.5) to reduce cost
+ As Prowl utilizes data crawled from the .xml finding aids, if a collection does not have a finding aid, Prowl will be unable to reference that collection
