			 ________    ________      ________      ___       __       ___          
			|\   __  \  |\   __  \    |\   __  \    |\  \     |\  \    |\  \         
			\ \  \|\  \ \ \  \|\  \   \ \  \|\  \   \ \  \    \ \  \   \ \  \        
 			 \ \   ____\ \ \   _  _\   \ \  \\\  \   \ \  \  __\ \  \   \ \  \       
 			  \ \  \___|  \ \  \\  \|   \ \  \\\  \   \ \  \|\__\_\  \   \ \  \____  
 			   \ \__\      \ \__\\ _\    \ \_______\   \ \____________\   \ \_______\
			    \|__|       \|__|\|__|    \|_______|    \|____________|    \|_______|

					     AI chatbot for the UWM Archives
                  			Matthew Cavenaile (and Claude/ChatGPT/Llama)                               
                          		       matt.cavenaile@gmail.com                                         
                                                                        
Purpose & Scope
============================================================================================================================
The purposes of this chatbot are to assist patrons with basic to intermediate level reference requests.
Prowl should be able to assist patrons in the following ways:
	- Answering general Archives questions (What are your hours? What kinds of records are available at the archives?)
	- Narrowing down archival search queries
	- Identifying specific archival finding aids
	- If Archives is the incorrect authority, Prowl should be able to
	  refer patrons to the correct UWM libraries resource
	

PROWL DevLog
============================================================================================================================
First step, create a python program to parse through the .xml data and export it to a .json file. This program turns the data into an easier machine readable format.

Started on Jupyter Labs.. prefer to work with notepad++

Missing EAD issue.. cross reference LOCATOR with eadlist.html possible?

Google Cloud DB Path: https://storage.googleapis.com/prowl_database/chroma_db/chroma.sqlite3


Prompt Engineering (Claude/ChatGPT Arcanum/Llama)
============================================================================================================================
+ XML Parsing
  If I show you an example of one of the XML documents, could you write a program that parses it and outputs to a JSON file? It
  needs:
  to be able to parse multiple files in a directory. The schema should be organized as: Title/Dates(<titleproper>), Call
  Number(<unitid>), Repository(<repository>), Abstract(<abstract>), Scope(<scopecontent>), Biology/History(<bioghist>), 
  Arrangement(<arrangement>), Physical Description(<physdesc>), Admin(<archdesc> and <descgrp>), and Contents(<dsc>). Note that the 
  () contain the corresponding XML element tags. Also note that I need ALL of the data from the <dsc> tags, including all of the 
  nested data. Please take as long as you need to come up with an ideal solution, I am no rush. And thank you! Attached is the 
  example XML document.

+ Embedding and initializing/storing in ChromaDB
  Excellent! I've been working on this for days and you've fixed my problems instantly! Next I'd like to embed and store the data
  in a vector database (ChromaDB). Could you help me with that?





To-do
============================================================================================================================
+ Possible to setup a notification system via AskArch if a patron would like to look at a collection? Clickable links in chat?
+ Future functionality - integrate w/ WTMJ collection?
+ Add text-bubbls to know when the bot is thinking



Security/Guardrails
============================================================================================================================




Pitching
============================================================================================================================
+ Affordability (Cost per query)
+ Versatility (Can be customized to fit the needs of the institution)
+ Sleek interface designed by graphic design professional
+ Security
+ Option to operate in-house(purchasable package that is customizable), or monthly subscription-based w/ 24/7 Support
+ Consultation services to tailor the bot to specific needs of the customer (simple or complex configs)
+ Ethical concerns?


Limitations
============================================================================================================================
+ It is worth noting that Prowl uses an older model of ChatGPT (3.5) to reduce cost
+ Unfortunately if we do not have a finding aid for the collection, Prowl will not be able to reference that collection
