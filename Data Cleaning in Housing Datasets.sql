CREATE DATABASE Nachville;
USE Nachville;

CREATE TABLE DataCleaning(
UniqueID INT(20),
ParcelID NVARCHAR(30),
LandUse	VARCHAR(30),
PropertyAddress	VARCHAR(30),
SaleDate	datetime,
SalePrice	INT(30),
LegalReference	INT(30),
SoldAsVacant CHAR(20),
OwnerName	CHAR(30),
OwnerAddress	VARCHAR(40),
Acreage	FLOAT(30),
TaxDistrict CHAR(30),
LandValue INT(30),
BuildingValue INT(30),	
TotalValue  INT(30),	
YearBuilt INT(20),	
Bedrooms INT(5),
FullBath INT(5),
HalfBath INT(5)
);

SELECT *
FROM nashvilledata;

-- Cleaning Data in SQL Queries

SELECT * 
FROM nashvilledata;

-- Standardized date format

-- SELECT SaleDate,convert (DATE,SaleDate)
-- FROM nashvilledata;
-- UPDATE nashvilledata
-- SET SaleDate=convert(DATE,SaleDate);

ALTER TABLE 
nashvilledata 
ADD SaleDateConverted  DATE;

-- UPDATE nashvilledata
-- SET SaleDateConverted=CONVERT(DATE,SaleDate);

-- Populate Property Address Data

SELECT PropertyAddress
FROM nashvilledata
WHERE PropertyAddress is NULL;


SELECT a.ParcelID,a.PropertyAddress,b.ParcelID,b.PropertyAddress ,ISNULL(a.PropertyAddress,b.PropertyAddress)
FROM nashvilledata a
JOIN nashvilledata b
ON a.ParcelID=b.ParcelID
AND
a.UniqueID<>b.UniqueID
WHERE a.PropertyAddress is NULL;

-- UPDATE a
-- SET PropertyAddress=ISNULL(a.PropertyAddress,b.PropertyAddress)
-- FROM nashvilledata a
-- JOIN nashvilledata b
-- ON a.ParcelID=b.ParcelID
-- AND
-- a.UniqueID<>b.UniqueID
-- WHERE a.PropertyAddress is NULL;

-- Breaking out Address into individual columns(like city,state etc.)

USE nachville;
SELECT 
SUBSTRING (propertyaddress,1,CHARINDEX(",",propertyaddress)) AS Address,
CHARINDEX(",",propertyaddress)
FROM nashvilledata;

-- owner address
SELECT
owneraddress FROM
nashvilledata;

SELECT
PARSENAME(REPLACE(owneradddress,",","-"),1)
FROM
nashvilledata;

-- Change Y or N in Yes or No in sold or vacant
SELECT *
FROM nashvilledata;

SELECT SoldAsVacant
FROM nashvilledata;

SELECT DISTINCT(SoldAsVacant),
COUNT(SoldAsVacant)
FROM nashvilledata;

SELECT SoldAsVacant
-- CASE WHEN SoldAsVacant="Y" THEN "YES"
-- WHEN SoldAsVacant="N" THEN "N"
-- ELSE
-- SoldAsVacant
-- END
From nashvilledata;

UPDATE nashvilledata
SET SoldAsVacant=CASE 
WHEN SoldAsVacant="Y" THEN "YES"
WHEN SoldAsVacant="N" THEN "NO"
ELSE
SoldAsVacant
END;

SELECT *
FROM nashvilledata;

ALTER TABLE nashvilledata
DROP COLUMN SaleDateConverted;

ALTER TABLE nashvilledata
DROP COLUMN OwnerAddress;