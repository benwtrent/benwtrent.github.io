---
layout: post
title: Via Kibana Maps 
tags: [geo, geospatial, elasticsearch, kibana, maps, gtfs]
published: true
---

## TL;DR

The Kibana Maps app is cool. Explore it yourself!

 - Download [Elasticsearch](https://www.elastic.co/downloads/elasticsearch) and [Kibana](https://www.elastic.co/downloads/kibana) 
 - [Importing gtfs data](https://github.com/benwtrent/gtfs-es-bootstrap) to your locally running cluster
 - [Get started](https://www.elastic.co/guide/en/kibana/7.0/maps-getting-started.html)


## Kibana Maps

With the v7.0.0 release<sup id="a1">[1](#f1)</sup>, the Kibana team released a beta version of the much anticipated Maps app. It allows visualizing complex and layered geospatial data with all the bells and whistles that is expected in a Kibana app<sup id="a2">[2](#f2)</sup>. With its release, I thought it would be fun to see how to visualize and store GTFS data.

## What is GTFS

General Transit Feed Specification (GTFS) provides a well known format for public transit systems and their related transit schedules. The specification provides formats for all common aspects of transit including: stops, times, routes, trips, calendar times, and fares<sup id="a3">[3](#f3)</sup>. These data are provided via static files that also optionally includes geospatial data. This enables  useful visualizations and search within transit system data. Google provides an [excellent and complete reference](https://developers.google.com/transit/gtfs/reference/). If you want to dig deeper, I recommend starting there. 

## Bringing GTFS into Elasticsearch

Elasticsearch has supported geospatial data for some time now. But, [recent improvements](https://www.elastic.co/elasticon/tour/2019/san-francisco/geospatial-advancements-in-elasticsearch-and-apache-lucene) have been impressive. This has enabled easier adoption of geospatial data and more responsive interactions. This made bringing Via's - my local transit company - GTFS data<sup id="a4">[4](#f4)</sup> into Elasticsearch a piece of cake.

### Formatting and pushing the data

In Elasticsearch, it is best to denormalize your data. Throughout the GTFS data set, there are references to `*_id` fields for joining data together. I opted for filling in those foreign keys with the actual object where applicable. As for formatting the geospatial data, Elasticsearch supports both [geo_point](https://www.elastic.co/guide/en/elasticsearch/reference/current/geo-point.html) and [geo_shape](https://www.elastic.co/guide/en/elasticsearch/reference/current/geo-shape.html) data mappings. For all data with simple `*_lat`/`*_lon` fields -  an example being `stops_lat` and `stops_lon` in `stops.txt` - I combined them to form a `geo_point` field. Parsing the provided `shapes.txt` proved to be slightly more involved. 

Each latitude (`shape_pt_lat`) and longitude (`shape_pt_lon`) point inside the defined shape are given on separate lines. Each of these lines are to be ordered by their related `shape_pt_sequence`. The best way to store this particular shape is with a [linestring geo_shape](https://www.elastic.co/guide/en/elasticsearch/reference/current/geo-shape.html#_ulink_url_http_geojson_org_geojson_spec_html_id3_linestring_ulink). I ran into two issues when parsing this data. One, `geo_shape` points are stored in `lon,lat` (X,Y) format - the opposite of `geo_point`. It took me way too long figure out why my paths were in the middle of Antarctica instead of San Antonio, TX. Two, inside Kibana Maps, support for [Well-Known Text (WKT)](http://docs.opengeospatial.org/is/12-063r5/12-063r5.html) is [currently broken](https://github.com/elastic/kibana/issues/32074). Consequently, all `geo_shapes` must be in the [GeoJSON](http://geojson.org/) format. 

Once I had all the data read, it was a simple matter of utilizing [elasticsearch-py](https://github.com/elastic/elasticsearch-py) to push it up. 

[You can see my whole script and the mappings on github](https://github.com/benwtrent/gtfs-es-bootstrap). Python is definitely not my daily language, so please excuse the mess :)

## Creating the Map

Kibana Maps comes built in with a default vector road map. I used this as my bottom layer. For each consecutive layer, I chose the `Documents` data source. 

![alt text](/assets/via_adding_layer.gif "Adding a new layer to Maps")

With each layer you have options to add a tooltip, adjust the size and color, and even "join" in some more external data to provide more meaning. 

Here is my completed map showing all of Via's routes in San Antonio, their Stops, Starts and Finishes.

![alt text](/assets/via_full.jpg "Via Kibana Maps").


---------------------------------------------------------------------------------------------

###### <b id="f1">1</b> [Elasticsearch download](https://www.elastic.co/downloads/elasticsearch) [Kibana download](https://www.elastic.co/downloads/kibana). Both need to be the "Basic" or Elastic Licensed version. It is free and open, but not OSS. [Subscription info](https://www.elastic.co/subscriptions)  [↩](#a1)

###### <b id="f2">2</b> [The maps docs](https://www.elastic.co/guide/en/kibana/7.0/maps.html) explains its current capabilities better than I ever could :)  [↩](#a2)

###### <b id="f3">3</b> For a more thorough overview [see Google's overview of static GTFS](https://developers.google.com/transit/gtfs/) [↩](#a3)

###### <b id="f4">4</b> [Data provided by VIA Metropolitan Transit](https://www.viainfo.net/developers-resources/) [↩](#a4)