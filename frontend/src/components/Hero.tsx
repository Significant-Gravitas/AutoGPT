import { ArrowRight, Sparkles } from 'lucide-react';
import { useState } from 'react';

interface HeroProps {
  onOpenDiagnostic: () => void;
}

export default function Hero({ onOpenDiagnostic }: HeroProps) {
  return (
    <section className="pt-24 pb-12 md:pt-32 md:pb-20 bg-gradient-to-br from-primary-50 via-white to-secondary-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          {/* Badge */}
          <div className="inline-flex items-center space-x-2 bg-white px-4 py-2 rounded-full shadow-sm mb-6">
            <Sparkles className="h-4 w-4 text-primary-600" />
            <span className="text-sm font-medium text-gray-700">
              Líderes en transformación digital con IA
            </span>
          </div>

          {/* Main heading */}
          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-gray-900 mb-6 leading-tight">
            NEUS impulsa la{' '}
            <span className="bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
              eficiencia empresarial
            </span>
            {' '}con Inteligencia Artificial
          </h1>

          {/* Subheading */}
          <p className="text-xl sm:text-2xl md:text-3xl text-gray-700 font-semibold mb-4">
            Reduce tus costos operativos hasta un{' '}
            <span className="text-primary-600">40%</span> y lleva tu negocio al siguiente nivel
          </p>

          {/* Description */}
          <p className="text-lg text-gray-600 max-w-3xl mx-auto mb-10">
            Transformamos empresas mediante soluciones de IA personalizadas. Desde capacitación hasta
            automatización completa de procesos, te acompañamos en cada paso de tu evolución digital.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={onOpenDiagnostic}
              className="group px-8 py-4 bg-gradient-to-r from-primary-600 to-secondary-600 text-white rounded-lg text-lg font-semibold hover:shadow-xl hover:scale-105 transition-all duration-200 flex items-center space-x-2"
            >
              <span>Diagnóstico Gratuito</span>
              <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </button>

            <button
              onClick={() => {
                const element = document.getElementById('servicios');
                element?.scrollIntoView({ behavior: 'smooth' });
              }}
              className="px-8 py-4 bg-white text-gray-700 border-2 border-gray-300 rounded-lg text-lg font-semibold hover:border-primary-600 hover:text-primary-600 transition-all duration-200"
            >
              Conocer servicios
            </button>
          </div>

          {/* Stats */}
          <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            <div className="p-6 bg-white rounded-xl shadow-sm">
              <div className="text-4xl font-bold bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent mb-2">
                40%
              </div>
              <div className="text-gray-600">Reducción de costos operativos</div>
            </div>

            <div className="p-6 bg-white rounded-xl shadow-sm">
              <div className="text-4xl font-bold bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent mb-2">
                8+
              </div>
              <div className="text-gray-600">Sectores especializados</div>
            </div>

            <div className="p-6 bg-white rounded-xl shadow-sm">
              <div className="text-4xl font-bold bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent mb-2">
                24/7
              </div>
              <div className="text-gray-600">Soporte y acompañamiento</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
