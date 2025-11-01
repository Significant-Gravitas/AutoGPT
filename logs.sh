#!/bin/bash
# Script para ver logs de todos los servicios

if [ -z "$1" ]; then
    echo "📋 Mostrando logs de todos los servicios..."
    docker-compose logs -f
else
    echo "📋 Mostrando logs de $1..."
    docker-compose logs -f $1
fi
