import { ShoppingCart, Heart, Truck, Building2, Scale, UserPlus, FileText, BookOpen } from 'lucide-react';

const sectors = [
  { name: 'Retail', icon: ShoppingCart, color: 'bg-blue-500' },
  { name: 'Salud', icon: Heart, color: 'bg-red-500' },
  { name: 'Supply Chain', icon: Truck, color: 'bg-green-500' },
  { name: 'Administración Pública', icon: Building2, color: 'bg-purple-500' },
  { name: 'Legal', icon: Scale, color: 'bg-yellow-600' },
  { name: 'Onboarding', icon: UserPlus, color: 'bg-indigo-500' },
  { name: 'Back-Office', icon: FileText, color: 'bg-cyan-500' },
  { name: 'Formación', icon: BookOpen, color: 'bg-pink-500' }
];

export default function Sectors() {
  return (
    <section id="sectores" className="py-20 bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            Sectores{' '}
            <span className="bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
              Especializados
            </span>
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Experiencia comprobada en múltiples industrias con soluciones adaptadas a cada sector
          </p>
        </div>

        {/* Sectors Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {sectors.map((sector, index) => {
            const Icon = sector.icon;
            return (
              <div
                key={index}
                className="group bg-white p-6 rounded-xl shadow-sm hover:shadow-xl transition-all duration-300 cursor-pointer"
              >
                <div className="flex flex-col items-center text-center space-y-3">
                  <div className={`${sector.color} p-4 rounded-lg group-hover:scale-110 transition-transform duration-300`}>
                    <Icon className="h-8 w-8 text-white" />
                  </div>
                  <span className="font-semibold text-gray-900 group-hover:text-primary-600 transition-colors">
                    {sector.name}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Additional Info */}
        <div className="mt-16 text-center">
          <div className="bg-white p-8 rounded-2xl shadow-sm max-w-3xl mx-auto">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Especialización por Industria
            </h3>
            <p className="text-gray-600 leading-relaxed">
              Cada sector tiene sus propios desafíos y regulaciones. Nuestro equipo cuenta con
              experiencia específica en cada industria, garantizando soluciones que no solo son
              técnicamente avanzadas, sino que también cumplen con los requisitos particulares de tu sector.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
